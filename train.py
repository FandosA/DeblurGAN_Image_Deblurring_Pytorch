import os
import torch
import numpy as np
import configargparse
from loss import GeneratorLoss, DiscriminatorLossWGANGP
from dataset_augmented import Dataset
from model import DeblurGAN, Discriminator
from utils import selectDevice, calculateAccuracy
from torch.utils.data import DataLoader


def updateLearningRate(arguments, old_lr):
    
    lrd = arguments.learning_rate / arguments.num_iters_decay
    new_lr = old_lr - lrd
    
    for param_group in optimizer_d.param_groups:
        param_group['lr'] = new_lr
        
    for param_group in optimizer_g.param_groups:
        param_group['lr'] = new_lr
    
    return new_lr


def train():
    
    min_val_loss_g = np.inf
    max_val_acc_d = 0
    bestEpoch_g = 0
    bestEpoch_d = 0

    train_losses_g = []
    val_losses_g = []
    
    train_accuracies_d = []
    val_accuracies_d = []
    train_losses_d = []
    
    old_learning_rate = args.learning_rate
    
    
    print('--------------------------------------------------------------')
    
    # Loop along epochs to do the training
    for i in range(1, args.num_iters + args.num_iters_decay + 1):
        
        print(f'EPOCH {i}')
        
        # Training loop
        train_loss_g = 0.0
        train_acc_d = 0.0
        train_loss_d = 0.0
        generator.train()
        discriminator.train()
        iteration = 1
        
        print('\nTRAINING')
        
        for image_real, image_blurred in train_loader:
            
            print('\rEpoch[' + str(i) + '/' + str(args.num_iters + args.num_iters_decay) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(train_loader)), end='')
            iteration += 1
            
            image_real, image_blurred = image_real.to(device), image_blurred.to(device)
            
            """
            ###########################################
            #              Discriminator              #
            ###########################################
            """
            image_fake = generator(image_blurred).detach()
            train_acc_d_temp = 0.0
            train_loss_d_temp = 0.0
            for _ in range(5):
                
                optimizer_d.zero_grad()
                
                real_logits = discriminator(image_real)
                fake_logits = discriminator(image_fake)
                loss_d = discriminator_loss(discriminator, real_logits, fake_logits, image_real, image_fake)
    
                loss_d.backward()
                optimizer_d.step()
                
                real_accuracy = calculateAccuracy(real_logits, torch.ones_like(real_logits))
                fake_accuracy = calculateAccuracy(fake_logits, torch.zeros_like(fake_logits))
                train_acc_d_temp += (real_accuracy + fake_accuracy).item() / 2
                train_loss_d_temp += loss_d.item()
                
            train_acc_d += (train_acc_d_temp / 5.0)
            train_loss_d += (train_loss_d_temp / 5.0)
            
            """
            ###########################################
            #                Generator                #
            ###########################################
            """
            optimizer_g.zero_grad()
            
            image_fake = generator(image_blurred)
            fake_logits = discriminator(image_fake)
            
            loss_g = generator_loss(image_fake, image_real, fake_logits)
            
            loss_g.backward()
            optimizer_g.step()
            
            train_loss_g += loss_g.item()
        
        
        # Validation loop
        val_loss_g = 0.0
        val_acc_d = 0.0
        generator.eval()
        discriminator.eval()
        iteration = 1

        print('')
        print('\nVALIDATION')
        
        with torch.no_grad():
            
            for image_real, image_blurred in validate_loader:
                
                print('\rEpoch[' + str(i) + '/' + str(args.num_iters + args.num_iters_decay) + ']: ' + 'iteration ' + str(iteration) + '/' + str(len(validate_loader)), end='')
                iteration += 1
                
                image_real, image_blurred = image_real.to(device), image_blurred.to(device)
                
                """
                ###########################################
                #              Discriminator              #
                ###########################################
                """
                image_fake = generator(image_blurred)
        
                real_logits = discriminator(image_real)
                fake_logits = discriminator(image_fake)
                
                real_accuracy = calculateAccuracy(real_logits, torch.ones_like(real_logits))
                fake_accuracy = calculateAccuracy(fake_logits, torch.zeros_like(fake_logits))
                val_acc_d += (real_accuracy + fake_accuracy).item() / 2
                
                """
                ###########################################
                #                Generator                #
                ###########################################
                """
                loss_g = generator_loss(image_fake, image_real, fake_logits)
                val_loss_g += loss_g.item()
    

        # Save loss and accuracy values
        train_accuracies_d.append(train_acc_d / len(train_loader))
        val_accuracies_d.append(val_acc_d / len(validate_loader))
        train_losses_d.append(train_loss_d / len(train_loader))
        train_losses_g.append(train_loss_g / len(train_loader))
        val_losses_g.append(val_loss_g / len(validate_loader))
        
        print("\n\nDiscriminator")
        print(f'- Train accuracy: {train_acc_d / len(train_loader):.3f}')
        print(f'- Validation accuracy: {val_acc_d / len(validate_loader):.3f}')
        print(f'- Train loss: {train_loss_d / len(train_loader):.3f}\n')
        print("Generator")
        print(f'- Train loss G: {train_loss_g / len(train_loader):.3f}')
        print(f'- Validation loss G: {val_loss_g / len(validate_loader):.3f}')
        
        # Update the learning rate as the paper deblurGAN says
        if i > args.num_iters:
            old_learning_rate = updateLearningRate(args, old_learning_rate)
        
        # Save the model every 10 epochs
        if i % 20 == 0:
            torch.save(generator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_g.pth"))
            torch.save(discriminator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_d.pth"))
            
        # Save the best generator model when loss decreases respect to the previous best loss
        if (val_loss_g / len(validate_loader)) < min_val_loss_g:
            # If first epoch, save model as best, otherwise, replace the previous best model with the current one
            if i == 1:
                torch.save(generator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best_g.pth"))
            else:
                os.remove(os.path.join(checkpoints_path, "checkpoint_" + str(bestEpoch_g) + "_best_g.pth"))
                torch.save(generator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best_g.pth"))
            
            print(f'\nValidation loss of Generator decreased: {min_val_loss_g:.3f} ---> {val_loss_g / len(validate_loader):.3f}\nModel saved')
                
            # Update parameters with the new best model
            min_val_loss_g = val_loss_g / len(validate_loader)
            bestEpoch_g = i
            
        # Save the best discriminator model when accuracy increases respect to the previous best accuracy
        if (val_acc_d / len(validate_loader)) > max_val_acc_d:
            # If first epoch, save model as best, otherwise, replace the previous best model with the current one
            if i == 1:
                torch.save(discriminator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best_d.pth"))
            else:
                os.remove(os.path.join(checkpoints_path, "checkpoint_" + str(bestEpoch_d) + "_best_d.pth"))
                torch.save(discriminator.state_dict(), os.path.join(checkpoints_path, "checkpoint_" + str(i) + "_best_d.pth"))
            
            print(f'\nValidation accuracy of Discriminator increased: {max_val_acc_d:.3f} ---> {val_acc_d / len(validate_loader):.3f}\nModel saved')
                
            # Update parameters with the new best model
            max_val_acc_d = val_acc_d / len(validate_loader)
            bestEpoch_d = i
            
        np.savetxt(os.path.join(log_dir_path, 'train_losses_g.txt'), np.array(train_losses_g))
        np.savetxt(os.path.join(log_dir_path, 'val_losses_g.txt'), np.array(val_losses_g))
        np.savetxt(os.path.join(log_dir_path, 'train_losses_d.txt'), np.array(train_losses_d))
        np.savetxt(os.path.join(log_dir_path, 'train_accs_d.txt'), np.array(train_accuracies_d))
        np.savetxt(os.path.join(log_dir_path, 'val_accs_d.txt'), np.array(val_accuracies_d))
            
        print("--------------------------------------------------------------")


if __name__ == "__main__":
    
    # Select parameters for training
    arg = configargparse.ArgumentParser()
    arg.add_argument('--json_file_train_path', type=str, default='images_paths_training.json', help='Train dataset file path.')
    arg.add_argument('--json_file_val_path', type=str, default='images_paths_validation.json', help='Validation dataset file path.')
    arg.add_argument('--log_dir', type=str, default='deblurGAN', help='Name of the folder to save the model.')
    arg.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    arg.add_argument('--num_workers', type=int, default=6, help='Number of threads to use in order to load the dataset.')
    arg.add_argument('--num_resblocks', type=int, default=9, help='Number of residual blocks for the generator.')
    arg.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    arg.add_argument('--num_iters', type=int, default=150, help='Number of epochs with the same learning rate.')
    arg.add_argument('--num_iters_decay', type=int, default=150, help='Epoch number to start decreasing the learning rate.')
    arg.add_argument('--lambda_generator', type=float, default=100, help='Weighting parameter in the generator for the perceptual loss.')
    arg.add_argument('--lambda_discriminator', type=float, default=10, help='Weighting parameter for the gradient penalty in the discriminator loss.')
    arg.add_argument('--GPU', type=bool, default=True, help='True to run the model in the GPU.')
    args = arg.parse_args()
    
    log_dir_path = args.log_dir + "_bs" + str(args.batch_size) + "_lr" + str(args.learning_rate) + "_numresblocks" + str(args.num_resblocks) + "_lambdaG" + str(args.lambda_generator) + "_lambdaD" + str(args.lambda_discriminator)
    assert not (os.path.isdir(log_dir_path)), 'The folder log_dir already exists, remove it or change its name'
    
    # Create folder to store checkpoints and training and validation losses and accuracies
    os.mkdir(log_dir_path)
    checkpoints_path = os.path.join(log_dir_path, 'checkpoints')
    os.mkdir(checkpoints_path)
    
    # Select device
    device = selectDevice(args)
            
    # Load dataset and dataloaders
    train_dataset = Dataset(args, train=True)
    validate_dataset = Dataset(args, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    total_images = len(train_dataset) + len(validate_dataset)
    print('Training images: ' + str(len(train_dataset)) + '/' + str(total_images))
    print('Validation images: ' + str(len(validate_dataset)) + '/' + str(total_images) + '\n')
    
    # Create models
    generator = DeblurGAN(n_resblocks=args.num_resblocks).to(device)
    discriminator = Discriminator().to(device)
    
    # Set up optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    # Create the loss functions
    generator_loss = GeneratorLoss(args.lambda_generator, device)
    discriminator_loss = DiscriminatorLossWGANGP(args.lambda_discriminator, device)
    
    # Train the model
    train()


