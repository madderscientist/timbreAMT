import torch
import shutil
import os

pthFolder = './'
bestFolder = './'

def save_checkpoint(state, is_best = False, filename='checkpoint.pth'):
    torch.save(state, os.path.join(pthFolder, filename))
    if is_best:
        shutil.copyfile(filename, os.path.join(bestFolder, 'best_'+filename))

def save_ckpt_template(epoch, model, min_loss, loss, optimizer, checkpoint_filename):
    # optimizer: 可以是 an optimizer 或 [optimizer1, optimizer2, ...]
    if isinstance(optimizer, (list, tuple)):
        optimizer_state = [opt.state_dict() for opt in optimizer]
    else:
        optimizer_state = optimizer.state_dict()
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'min_loss': min_loss,
        'avg_loss': loss,   # 一般是验证集上的损失
        'optimizer': optimizer_state,
    }, loss < min_loss, filename = checkpoint_filename)

def load_ckpt_template(model, optimizer, checkpoint_path):
    device = next(model.parameters()).device
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        if isinstance(optimizer, (list, tuple)):
            for opt, state in zip(optimizer, checkpoint['optimizer']):
                opt.load_state_dict(state)
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        min_loss = checkpoint['min_loss']
        avg_loss = checkpoint['avg_loss']
        epoch_now = checkpoint['epoch']
        print(f"Checkpoint loaded from '{checkpoint_path}'\nepoch: {epoch_now}, loss: {avg_loss}, min_loss: {min_loss}")
        return (min_loss, avg_loss, epoch_now)
    else:
        print(f"Checkpoint file '{checkpoint_path}' does not exist.")
        return False