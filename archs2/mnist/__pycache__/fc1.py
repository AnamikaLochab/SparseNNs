o
    �F6g�  �                   @   s*   d dl Z d dlmZ G dd� dej�ZdS )�    Nc                       s&   e Zd Zd� fdd�	Zdd� Z�  ZS )�fc1�
   c              
      sN   t t| ���  t�t�dd�tjdd�t�dd�tjdd�t�d|��| _d S )Ni  i,  T)�inplace�d   )�superr   �__init__�nn�
Sequential�Linear�ReLU�
classifier)�self�num_classes��	__class__� �R/u/antor/u12/alochab/cs578/Lottery-Ticket-Hypothesis-in-Pytorch/archs/mnist/fc1.pyr      s   





�zfc1.__init__c                 C   s   t �|d�}| �|�}|S )N�   )�torch�flattenr   )r   �xr   r   r   �forward   s   
zfc1.forward)r   )�__name__�
__module__�__qualname__r   r   �__classcell__r   r   r   r   r      s    
r   )r   �torch.nnr   �Moduler   r   r   r   r   �<module>   s    