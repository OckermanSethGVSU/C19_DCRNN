import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from dcrnn_model import DCRNNModel
from loss import masked_mae_loss

from dask.distributed import LocalCluster
from dask.distributed import Client
from dask.array.lib.stride_tricks import sliding_window_view
from dask.distributed import wait as Wait
from dask.delayed import delayed
import dask.array as da
import dask.dataframe as dd
from dask_pytorch_ddp import dispatch, results
import dask

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, Sampler

import uuid
import pandas as pd
import math
import threading

class IndexTrainDataset(Dataset):
    def __init__(self,x, data, lazy=False):
         self.x = x 
         self.data = data
         self.lazy = lazy 
         self.total = 0
        
        
    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):

        if self.lazy:
            y_start = idx + 12
            y2 = y_start + 12

            # t1 = time.time()
            x=  torch.from_numpy(self.data[idx:y_start,...].compute())
            y = torch.from_numpy(self.data[y_start:y2,...].compute())
            # t2 = time.time()
            # self.total += (t2-t1)
            
            return x,y
            
        else:
            
            # t1 = time.time()
            y_start = idx + 12
            
            x=  torch.from_numpy(self.data[idx:y_start,...])
            y = torch.from_numpy(self.data[y_start:y_start + 12,...])
            # t2 = time.time()
            # self.total += (t2-t1)

            return x,y
            
class IndexValDataset(Dataset):
    def __init__(self,x, data, lazy=False):
         self.x = x 
         self.data = data
         self.lazy = lazy  

    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        y_start = idx + 12
        if self.lazy:
            y_start = idx + 12
            
            return torch.from_numpy(self.data[idx:y_start,...].compute()), torch.from_numpy(self.data[y_start:y_start + 12,...].compute())
            
        else:
            y_start = idx + 12
            return torch.from_numpy(self.data[idx:y_start,...]), torch.from_numpy(self.data[y_start:y_start + 12,...])
                    

class DaskTrainDataset(Dataset):
    def __init__(self,x, y, lazy_batching=False):
         self.x = x 
         self.y = y 
         self.lb = lazy_batching
         self.total = 0
        
    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.lb:
            # t1 = time.time()

            x_tensor = torch.from_numpy(self.x[idx].compute())
            y_tensor = torch.from_numpy(self.y[idx].compute())
            # t2 = time.time()
            # self.total += (t2-t1)
            return x_tensor, y_tensor
        # t1 = time.time()
        x_tensor = torch.from_numpy(self.x[idx])
        y_tensor = torch.from_numpy(self.y[idx])
        # t2 = time.time()
        # self.total += (t2-t1)
        return x_tensor, y_tensor
        

class DaskValDataset(Dataset):
    def __init__(self, x,y, lazy_batching=False):
         self.x = x
         self.y = y
         self.lb = lazy_batching

    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.lb:
            x_tensor = torch.from_numpy(self.x[idx].compute())
            y_tensor = torch.from_numpy(self.y[idx].compute())
           
            return x_tensor, y_tensor
 
        return self.x[idx], self.y[idx]

import psutil
def monitor_memory_usage():
    while True:
        system_memory = psutil.virtual_memory()
        print(f"Used memory: {system_memory.used}")
        print()
        time.sleep(0.25)

def my_train(x_train=None, y_train=None, x_val=None, y_val=None, mean=None, std=None,
             graph_data=None, data_size=None,
            start_time=None, 
            train_dict=None, model_dict=None, data_dict=None):
            worker_rank = int(dist.get_rank())
            
            if train_dict['mode'] == 'dist':
                device = f"cuda:{worker_rank % 4}"
                torch.cuda.set_device(worker_rank % 4)
            else:
                device = None

            from utils import load_graph_data
            sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(model_dict['graph_pkl_filename'])
            

            index_lazy = False
            if train_dict['impl'].lower() == 'dask-index':
                
                x_i = np.arange(data_size)
                num_samples = x_i.shape[0]
                num_test = round(num_samples * 0.2)
                num_train = round(num_samples * 0.7)
                num_val = num_samples - num_test - num_train

            

                x_train = x_i[:num_train]
                x_val = x_i[num_train: num_train + num_val]
                data = graph_data
                index_lazy = True
            
            
            if train_dict['impl'].lower() == 'index':
                t1 = time.time()
                
                if "covid" in train_dict['h5']:
                    data = np.load(train_dict['h5'])
                    
                    num_samples, num_nodes, _ = data.shape
                    
                    x_offsets = np.sort(np.arange(-11, 1, 1))
                    y_offsets = np.sort(np.arange(1, 13, 1))
                    
                else:
                    df = pd.read_hdf(train_dict['h5'], key=train_dict['h5key'])
                    df = df.astype('float32')
                    
                    if 'bay' in train_dict['h5']:
                        pass
                    else:
                        df.index.freq='5min'  # Manually assign the index frequency
                    df.index.freq = df.index.inferred_freq

                    x_offsets = np.sort(
                        np.concatenate((np.arange(-11, 1, 1),))
                    )
                    
                    # Predict the next one hour
                    y_offsets = np.sort(np.arange(1, 13, 1))

                    num_samples, num_nodes = df.shape
                    data = np.expand_dims(df.values, axis=-1)

                    add_time_in_day = True
                    add_day_in_week = False
                    
                    data_list = [data[:]]
                
                    if add_time_in_day:
                        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
                        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
                        data_list.append(time_in_day)
                    


                    data = np.concatenate(data_list, axis=-1)
                
                x, y = [], []
                # t is the index of the last observation.
                min_t = abs(min(x_offsets))
                max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
                x_i = np.arange(x_offsets[0] + max_t)

                concat = False
                # concat method
                bin_len = round(x_i.shape[0] * 0.7) + 11
                if concat:
                    ascending = np.arange(1, 12)
                    descending = np.arange(11, 0, -1)
                    remaining_elements = bin_len - (len(ascending) + len(descending))

                    repeat_12 = np.full(remaining_elements, 12)
                    
                    # Step 3: Concatenate the sequences
                    bin = np.concatenate((ascending, repeat_12, descending))

                else: 
                    # pre-alloc
                    bin = np.empty(bin_len, dtype=int)
                    
                    # Step 2: Fill in the ascending part [1, 2, ..., 12]
                    bin[:12] = np.arange(1, 13)
                    # Step 3: Calculate how many times to repeat 12
                    remaining_elements = bin_len - 24
                    bin[12:13+remaining_elements] = 12
                    
                    # Step 4: Fill in the descending part [11, 10, ..., 1]
                    bin[13+remaining_elements:] = np.arange(11, 0, -1)
                

                num_samples = x_i.shape[0]
                num_test = round(num_samples * 0.2)
                num_train = round(num_samples * 0.7)
                num_val = num_samples - num_test - num_train

                

                x_train = x_i[:num_train]
                x_val = x_i[num_train: num_train + num_val]
                
                cutoff = bin.shape[0]
                
                total_entries = x_train.shape[0] * 12 * num_nodes
            
                mean = (data[: cutoff,..., 0].sum(axis=1) * bin).sum() / total_entries
                std = np.sqrt(  ((np.square(data[: cutoff, ..., 0] - mean)).sum(axis=1) * bin ).sum() / total_entries )
                

                data[..., 0] = (data[..., 0] - mean) / std
                
                t2 = time.time()
                if worker_rank == 0:
                    if os.path.exists("stats.txt"):
                        with open("stats.txt", "a") as file:
                                file.write(f"pre_processing_time: {t2 - t1}\n")
                    else:
                        with open("stats.txt", "a") as file:
                            file.write(f"pre_processing_time: {t2 - t1}\n")
                    
            train_start = time.time()
            if not os.path.exists("logs/"):
                os.makedirs("logs/", exist_ok=True)
            if not os.path.exists("logs/info.log"):
                with open("logs/info.log", 'w') as file:
                    file.write('')
            log_dir = "logs/"
            writer = SummaryWriter('runs/' + log_dir)

            log_level =  "INFO"
            logger = utils.get_logger(log_dir, __name__, 'info.log', level=log_level)
            


            if train_dict['impl'].lower() == 'index' or train_dict['impl'].lower() == 'dask-index':
                train_dataset = IndexTrainDataset(x_train,data, lazy=index_lazy)
                val_dataset = IndexValDataset(x_val, data, lazy=index_lazy)


            elif train_dict['impl'].lower() == 'dask':
                x_train = x_train.compute()
                y_train = y_train.compute()
                x_val = x_val.compute()
                y_val = y_val.compute()

                train_dataset = DaskTrainDataset(x_train, y_train, lazy_batching=False)
                val_dataset = DaskValDataset(x_val, y_val, lazy_batching=False)

            elif train_dict['impl'].lower() == 'lazydask':
                train_dataset = DaskTrainDataset(x_train, y_train, lazy_batching=True)
                val_dataset = DaskValDataset(x_val, y_val, lazy_batching=True)

            train_sampler = DistributedSampler(train_dataset, 
                                num_replicas=train_dict['npar'], 
                                rank=worker_rank, 
                                shuffle=True, 
                                drop_last=True)
            val_sampler = DistributedSampler(val_dataset, 
                                num_replicas=train_dict['npar'], 
                                rank=worker_rank, 
                                shuffle=True, 
                                drop_last=True)
            
            
            train_loader = DataLoader(train_dataset, batch_size=data_dict['batch_size'], sampler=train_sampler, shuffle=False, drop_last=True)
            train_per_epoch = (x_train.shape[0] // data_dict['batch_size']) // train_dict['npar']
            val_loader = DataLoader(val_dataset, batch_size=data_dict['batch_size'], sampler=val_sampler, shuffle=False, drop_last=True)
            val_per_epoch = (x_val.shape[0] // data_dict['batch_size']) // train_dict['npar']
            


            scaler = utils.StandardScaler(mean=mean, std=std)

            
            
            model = DCRNNModel(adj_mx, logger, **model_dict)
            model = DDP(model, gradient_as_bucket_view=True).to(device)
            optimizer = torch.optim.Adam(model.module.parameters(), lr=train_dict['base_lr'], eps=train_dict['epsilon'])

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_dict['steps'],
                                                                gamma=train_dict['lr_decay_ratio'])
            
            

            logger.info('Start training ...')
            epochs = train_dict['epochs']
            start_epoch = 0
            batches_seen = 0
            
            if worker_rank == 0:
                print("Model created successfully; About to begin epochs", flush=True)

                if os.path.exists("per_epoch_stats.txt"):
                    pass
                else:
                    with open("per_epoch_stats.txt", "w") as file:

                            file.write(f"epoch,batches_seen,batches_seen,train_loss,val_loss,lr\n")
            
            overall_t_loss = []
            overall_v_loss = []
            for epoch_num in range(start_epoch, epochs):
                
                model = model.train()
                train_sampler.set_epoch(epoch_num)
                val_sampler.set_epoch(epoch_num)
                
                losses = []
                t1 = time.time()

                for i, (x, y) in enumerate(train_loader):

                    if worker_rank == 0:
                        print(f"\rEpoch {epoch_num} train batch {i + 1}/{train_per_epoch}", flush=True, end="")
                    train_dataset.total = 0

                    
                    optimizer.zero_grad()

                    
                    x = x.to(device).float()
                    y = y.to(device).float()
                    x = x.permute(1, 0, 2, 3)
                    y = y.permute(1, 0, 2, 3)
                    batch_size = x.size(1)
                    
                    x = x.view(model_dict['seq_len'], batch_size, model_dict['num_nodes'] * \
                               model_dict['input_dim'])
                    
                    y = y[..., :model_dict['output_dim']].view(model_dict['horizon'], batch_size,
                                                    model_dict['num_nodes'] * model_dict['output_dim'])
                    
                    output = model(x, y, batches_seen)
                    
                    if batches_seen == 0:
                        # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                        optimizer = torch.optim.Adam(model.module.parameters(), lr=train_dict['base_lr'], eps=train_dict['epsilon'])
                        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_dict['steps'],
                                                                    gamma=train_dict['lr_decay_ratio'])                
                    y_true = scaler.inverse_transform(y)
                    y_predicted = scaler.inverse_transform(output)

                   
                    loss = masked_mae_loss(y_predicted, y_true)

                    
                    logger.debug(loss.item())

                    losses.append(loss.item())

                    batches_seen += 1
                    loss.backward()

                    # gradient clipping - this does it in place
                    torch.nn.utils.clip_grad_norm_(model.module.parameters(), train_dict['max_grad_norm'])
                    
                    optimizer.step()
                    

                
                logger.info("epoch complete")
                lr_scheduler.step()
                logger.info("evaluating now!")
                
                with torch.no_grad():
                    model = model.eval()


                    val_losses = []

                    y_truths = []
                    y_preds = []

                    for i, (x, y) in enumerate(val_loader):
                        if worker_rank == 0:
                            print(f"\rEpoch {epoch_num} val batch {i + 1}/{val_per_epoch}", flush=True, end="")
                    
                        x = x.to(device).float()
                        y = y.to(device).float()
                        logger.debug("X: {}".format(x.size()))
                        logger.debug("y: {}".format(y.size()))
                        x = x.permute(1, 0, 2, 3)
                        y = y.permute(1, 0, 2, 3)
                        batch_size = x.size(1)
                        x = x.view(model_dict['seq_len'], batch_size, model_dict['num_nodes'] * \
                                model_dict['input_dim'])
                        
                        y = y[..., :model_dict['output_dim']].view(model_dict['horizon'], batch_size,
                                                        model_dict['num_nodes'] * model_dict['output_dim'])

                        output = model(x)
                        y_true = scaler.inverse_transform(y)
                        y_predicted = scaler.inverse_transform(output)
                        loss = masked_mae_loss(y_predicted, y_true)
                        val_losses.append(loss.item())

                        y_truths.append(y.cpu())
                        y_preds.append(output.cpu())
                        
                        

                    val_loss = np.mean(val_losses)
                    overall_v_loss.append(val_loss)
                    writer.add_scalar('{} loss'.format('val'), val_loss, batches_seen)

                t2 = time.time()

                writer.add_scalar('training loss',
                                        np.mean(losses),
                                        batches_seen)

                if worker_rank == 0:
                    overall_t_loss.append(np.mean(losses))
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                            '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                            np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                            (t2 - t1))
                    logger.info(message)
                    print(message)

                    
                    if not os.path.exists('models/'):
                        os.makedirs('models/')

                    checkpoint = {
                        'epoch': epoch_num,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                        'batches_seen' : batches_seen
                    }


                    # print(self.state_dict())
                    torch.save(checkpoint, 'model_%d.pth' % epoch_num)
                    print(f"\nCheckpoint saved to {'model_%d.pth' % epoch_num}")
                    
                    with open("per_epoch_stats.txt", "a") as file:
                        # file.write(f"epoch, per_epoch_runtime, train_loss, val_loss, val_rmse, val_mape\n")
                        file.write(f"{epoch_num},{batches_seen},{t2 - t1},{np.mean(losses)},{val_loss},{lr_scheduler.get_last_lr()}\n")

            end_time = time.time()
            if worker_rank == 0:

                with open("stats.txt", "a") as file:
                    file.write(f"training_time: {end_time - train_start}\n")
                    file.write(f"total_time: {end_time - start_time}\n")

                    file.write(f"train_opt_loss: {min(overall_t_loss)}\n")
                    # file.write(f"train_opt_rmse: {min(overall_t_rmse)}\n")
                    # file.write(f"train_opt_mape: {min(overall_t_mape)}\n")

                    file.write(f"val_opt_loss: {min(overall_v_loss)}\n")

                

class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder
        self._epoch_num = self._train_kwargs.get('epoch', 0)
                
    
    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch, optimizer = None, lr_scheduler = None, batches_seen = None):
        """
        Save the model checkpoint to a file.

        :param filepath: Path to the file where the checkpoint will be saved.
        :param epoch: Current epoch number.
        :param optimizer: Optimizer state to save along with model parameters (optional).
        """
        if not os.path.exists('models/'):
            os.makedirs('models/')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.dcrnn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'batches_seen' : batches_seen
        }


        # print(self.state_dict())
        torch.save(checkpoint, 'model_%d.pth' % epoch)
        print(f"\ndaCheckpoint saved to {'model_%d.pth' % epoch}")


        return 'models/epo%d.tar' % epoch

    # TODO fix
    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)

        return self._train(**kwargs)

    


    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        
        def readPD():
            import math
            df = pd.read_hdf(self._train_kwargs['h5'], key=self._train_kwargs['h5key'])
            df = df.astype('float32')
            
            
            if 'bay' in self._train_kwargs['h5']:
                pass
            else:
                pass
            
            df.index.freq = df.index.inferred_freq
            return df
        
        # steps is used in learning rate - will see if need to use it?
        def get_numeric_part(filename):
            return int(filename.split("_")[1].split(".")[0])
        
        # monitor_thread = threading.Thread(target=monitor_memory_usage)
        # monitor_thread.daemon = True  # Daemonize thread to exit when the main program exits
        # # monitor_thread.start()

        if self._train_kwargs['load_path'] == "auto":
            current_dir = os.getcwd()
            files = os.listdir(current_dir)
            rel_files = [f for f in files if ".pth" in f]

            if len(rel_files) > 0:
                sorted_filenames = sorted(rel_files, key=get_numeric_part)
                self._train_kwargs['load_path'] = sorted_filenames[-1]
            else:
                self._train_kwargs['load_path'] = None



        start_time = time.time()
        if kwargs['mode'] == 'local':
            cluster = LocalCluster(n_workers=kwargs['npar'])
            client = Client(cluster)
        elif kwargs['mode'] == 'dist':
            client = Client(scheduler_file = f"cluster.info")
        else:
            print(f"{kwargs['mode']} is not a valid mode; Please enter mode as either 'local' or 'dist'")
            exit()
        
        x_train=None
        y_train=None
        x_val=None
        y_val=None
        mean=None
        std=None
        data_length=None
        data = None
        
        if self._train_kwargs['impl'].lower() == "dask":
            print("Dask implementation in use ", flush=True)
            
        elif self._train_kwargs['impl'].lower() == "lazydask":
            print("Lazy-dask-batching implementation in use ", flush=True)
            
        elif self._train_kwargs['impl'].lower() == "index":
            print("Index-batching implementation in use ", flush=True)
        
        elif self._train_kwargs['impl'].lower() == "dask-index":
            print("Dask-Index-batching implementation in use ", flush=True)
            
        else:
            print(f"{self._train_kwargs['impl'].lower()} is not a valid option; Please enter impl as either 'index','dask', or 'lazyDask")
            exit()
        
        if self._train_kwargs['impl'].lower() == "dask-index":
            
            
            t1 = time.time()
                
            if "covid" in self._train_kwargs['h5']:
                numpy_arr = np.load(self._train_kwargs['h5'])
                
                num_samples, num_nodes, _ = numpy_arr.shape
                
                x_offsets = np.sort(np.arange(-11, 1, 1))
                y_offsets = np.sort(np.arange(1, 13, 1))
                data = dask_array = da.from_array(numpy_arr)
                data = data.rechunk("auto")
                
            else:

                dfs = delayed(readPD)()
                df = dd.from_delayed(dfs)
                
                min_val_loss = float('inf')
                

                num_samples, num_nodes = df.shape

                num_samples = num_samples.compute()
                
                x_offsets = np.sort(np.arange(-11, 1, 1))
                y_offsets = np.sort(np.arange(1, 13, 1))
                
                data1 =  df.to_dask_array(lengths=True)
                data1 = da.expand_dims(data1, axis=-1)
                data1 = data1.rechunk("auto")

                data2 = da.tile((df.index.values.compute() - df.index.values.compute().astype("datetime64[D]")) / np.timedelta64(1, "D"), [1, num_nodes, 1]).transpose((2, 1, 0))
                data2 = data2.rechunk((data1.chunks))
                data = da.concatenate([data1, data2], axis=-1)


            x, y = [], []
            # t is the index of the last observation.
            min_t = abs(min(x_offsets))
            max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
            x_i = np.arange(x_offsets[0] + max_t)
            data_length = x_offsets[0] + max_t

            concat = False
            # concat method
            bin_len = round(x_i.shape[0] * 0.7) + 11
            if concat:
                ascending = np.arange(1, 12)
                descending = np.arange(11, 0, -1)
                remaining_elements = bin_len - (len(ascending) + len(descending))
                repeat_12 = np.full(remaining_elements, 12)
                
                # Step 3: Concatenate the sequences
                bin = np.concatenate((ascending, repeat_12, descending))

            else: 
                # pre-alloc
                bin = np.empty(bin_len, dtype=int)
                
                # Step 2: Fill in the ascending part [1, 2, ..., 12]
                bin[:12] = np.arange(1, 13)
                # Step 3: Calculate how many times to repeat 12
                remaining_elements = bin_len - 24
                bin[12:13+remaining_elements] = 12
                
                # Step 4: Fill in the descending part [11, 10, ..., 1]
                bin[13+remaining_elements:] = np.arange(11, 0, -1)
            

            num_samples = x_i.shape[0]
            num_test = round(num_samples * 0.2)
            num_train = round(num_samples * 0.7)
            num_val = num_samples - num_test - num_train

            

            x_train = x_i[:num_train]
            x_val = x_i[num_train: num_train + num_val]

            cutoff = bin.shape[0]
            
            total_entries = x_train.shape[0] * 12 * num_nodes
        
            mean = (data[: cutoff,..., 0].sum(axis=1) * bin).sum() / total_entries
            std = da.sqrt(  ((da.square(data[: cutoff, ..., 0] - mean)).sum(axis=1) * bin ).sum() / total_entries )
            

            data[..., 0] = (data[..., 0] - mean) / std
            data, mean, std,  = client.persist([data, mean, std])
            
            Wait([data, mean, std])
            mean = mean.compute()
            std = std.compute()
            
            pre_end = time.time()
            print(f"Preprocessing complete in {pre_end - t1}; Training Starting")
            
            if os.path.exists("stats.txt"):
                with open("stats.txt", "a") as file:
                        file.write(f"pre_processing_time: {pre_end - t1}\n")
            else:
                with open("stats.txt", "a") as file:
                    file.write(f"pre_processing_time: {pre_end - t1}\n")
            

            
            
        if self._train_kwargs['impl'].lower() == 'lazydask' or self._train_kwargs['impl'].lower() == 'dask':
            t1 = time.time()
            
            if "covid" in self._train_kwargs['h5']:
                numpy_arr = np.load(self._train_kwargs['h5'])
                
                num_samples, num_nodes, _ = numpy_arr.shape
                
                x_offsets = np.sort(np.arange(-11, 1, 1))
                y_offsets = np.sort(np.arange(1, 13, 1))
                memmap_array  = da.from_array(numpy_arr)
                memmap_array = memmap_array.rechunk("auto")
            else:
                dfs = delayed(readPD)()
                df = dd.from_delayed(dfs)
                min_val_loss = float('inf')
                

                num_samples, num_nodes = df.shape

                num_samples = num_samples.compute()
                
                x_offsets = np.sort(np.arange(-11, 1, 1))
                y_offsets = np.sort(np.arange(1, 13, 1))
                
                data1 =  df.to_dask_array(lengths=True)
                data1 = da.expand_dims(data1, axis=-1)
                data1 = data1.rechunk("auto")

                data2 = da.tile((df.index.values.compute() - df.index.values.compute().astype("datetime64[D]")) / np.timedelta64(1, "D"), [1, num_nodes, 1]).transpose((2, 1, 0))
                data2 = data2.rechunk((data1.chunks))
                
                memmap_array = da.concatenate([data1, data2], axis=-1)
                  
                del df
                del data1 
                del data2

            
            min_t = abs(min(x_offsets))
            max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
            total = max_t - min_t

            window_size = 12
            original_shape = memmap_array.shape

            
            # Define the window shape
            window_shape = (window_size,) + original_shape[1:]  # (12, 207, 2)

            # Use sliding_window_view to create the sliding windows
            # sliding_windows = np.lib.stride_tricks.sliding_window_view(memmap_array, window_shape).squeeze()
            sliding_windows = sliding_window_view(memmap_array, window_shape).squeeze()
            
            
            x = x_offsets[0] + max_t
            min_t = abs(min(x_offsets))
            max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
            x_i = np.arange(x)
            import random

          
            
            x_array = sliding_windows[:total]
            y_array = sliding_windows[window_size:]

            
            del memmap_array
            del sliding_windows





            num_samples = x_array.shape[0]
            num_test = round(num_samples * 0.2)
            num_train = round(num_samples * 0.7)
            num_val = num_samples - num_test - num_train

            

            x_train = x_array[:num_train]
            y_train = y_array[:num_train]

            mean = x_train[..., 0].mean()
            std = x_train[..., 0].std()
            

            x_train[..., 0] = (x_train[..., 0] - mean) / std
            y_train[..., 0] = (y_train[..., 0] - mean) / std
            
            


            x_val = x_array[num_train: num_train + num_val]
            y_val = y_array[num_train: num_train + num_val]


            x_val[..., 0] = (x_val[..., 0] - mean) / std
            y_val[..., 0] = (y_val[..., 0] - mean) / std
            
            
            mean, std, x_train, y_train, x_val, y_val = client.persist([mean, std, x_train, y_train, x_val, y_val])
            
            
            
            Wait([mean, std, x_train, y_train, x_val, y_val])
            

            mean = mean.compute()
            std = std.compute()
            
            
            pre_end = time.time()
            print(f"Preprocessing complete in {pre_end - t1}; Training Starting")
            
            if os.path.exists("stats.txt"):
                with open("stats.txt", "a") as file:
                        file.write(f"pre_processing_time: {pre_end - t1}\n")
            else:
                with open("stats.txt", "a") as file:
                    file.write(f"pre_processing_time: {pre_end - t1}\n")

            
            del x_array
            del y_array
        
        if kwargs['mode'] == "dist":
            for f in ['utils.py', 'dcrnn_cell.py', 'dcrnn_model.py', 'loss.py', 'index_based.py']:
                client.upload_file(f)
        
        
        futures = dispatch.run(client, my_train,
                               x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, mean=mean, std=std,
                               graph_data = data, data_size = data_length,
                               start_time = start_time,
                               train_dict= self._train_kwargs, model_dict=self._model_kwargs, data_dict=self._data_kwargs,
                               backend="gloo")
        key = uuid.uuid4().hex
        rh = results.DaskResultsHandler(key)
        rh.process_results(".", futures, raise_errors=False)
        client.shutdown()
        
        
        
        

    def _prepare_data(x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
