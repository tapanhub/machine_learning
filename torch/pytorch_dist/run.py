"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def runallreduce(rank, size):
    """ Simple point-to-point communication. """
    print(f"run called with rank={rank} size={size} pid={os.getpid()}")
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])


def runnonblocking(rank, size):
    """ Distributed function to be implemented later. """
    print(f"run called with rank={rank} size={size} pid={os.getpid()}")
    tensor = torch.zeros(2,2)
    req = None
    if rank == 0:
        tensor += 4
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print(f'Rank {rank} started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print(f'Rank {rank} started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])


def runblocking(rank, size):
    """ Distributed function to be implemented later. """
    print(f"run called with rank={rank} size={size} pid={os.getpid()}")
    tensor = torch.zeros(2,2)
    if rank == 0:
        tensor += 4
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, runallreduce))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
