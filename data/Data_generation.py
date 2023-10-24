import torch

class DataGenerator:
    def __init__(self, args):
        self.args = args
        # Set device
        if args.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # variance of complex AWGN
        self.r2 = torch.tensor(args.r2) 

    def doa_generator(self):
        """
        Output DoA [sample, k] matrix, 
        where each row is a sample of k DOAs,
        sorted in ascending order.
        """
        # Generate a tensor of angles from -doa_gt_range to doa_generate_range with doa_gt_increment increments
        all_angles = torch.arange(-self.args.doa_gt_range, self.args.doa_gt_range + self.args.doa_gt_increment, self.args.doa_gt_increment).to(self.device)
        x_dire = torch.zeros(self.args.sample, self.args.k, device=self.device)
        for sample_index in range(self.args.sample):
            """Generates k DOAs with a specified gap."""
            # Shuffle the angles
            shuffled_angles = all_angles[torch.randperm(all_angles.size(0))]

            selected_angles = torch.tensor([], device=self.device)
            for angle in shuffled_angles:
                if all(torch.abs(angle - selected_angles) >= self.args.gap):
                    selected_angles = torch.cat((selected_angles, angle.unsqueeze(0)))
                    if len(selected_angles) == self.args.k:
                        break
            selected_angles.sort()
            x_dire[sample_index] = selected_angles

        return x_dire
    

    def ULA_narrow(self):
        A = torch.zeros(self.args.n, self.args.m, dtype=torch.cfloat, device=self.device)

        # Generating i and s values as tensors
        i_values = torch.arange(self.args.n, dtype=torch.float32, device=self.device)
        s_values = torch.arange(self.args.m, dtype=torch.float32, device=self.device)

        # Calculate the angle matrix using broadcasting
        angles = torch.sin(s_values * (torch.pi / self.args.m) - torch.pi / 2)
        x = -1j * torch.pi * torch.outer(i_values, angles)

        # Compute the exponential
        A = torch.exp(x)

        return A
    
    #### generate dictionary matrices for each little windows ####
    def ULA_refine(self, centers, half_window_size): 
        
        center_num = len(centers) # number of centers    
        A = torch.zeros(center_num, self.args.n, self.args.m_SF, dtype = torch.cfloat, device=self.device) # dictionary matrices
        A_H = torch.zeros(center_num, self.args.m_SF, self.args.n, dtype = torch.cfloat, device=self.device) # transpose of dictionary matrices
        
        for ct in range(center_num):
            # b1 and b2 are the left and right boundaries of the window  
            b1 = centers[ct] - half_window_size
            b2 = centers[ct] + half_window_size
            b1 = b1 * torch.pi/180 # convert deg to rad
            b2 = b2 * torch.pi/180 # convert deg to rad
            angles = torch.linspace(b1, b2, self.args.m_SF, device=self.device)
            angles = torch.sin(angles)
            i_values = torch.arange(self.args.n, dtype=torch.float32, device=self.device)
            x = -1j * torch.pi * torch.outer(i_values, angles)
            # Compute the exponential
            A[ct, :, :] = torch.exp(x)
            # Compute the conjugate transpose
            A_H[ct, :, :] = A[ct, :, :].conj().T
        
        return A, A_H


    def steeringmatrix(self, doa):
        """
        Generates samples of steering matrices given DOA.
        input: doa, tensor of size [sample, k].
        output: steering matrix, tensor of size [sample, n, k].
        """
        STM_A = torch.zeros(self.args.sample, self.args.n, self.args.k, dtype=torch.cfloat, device=self.device)

        for sample_index in range(self.args.sample):
            s_values = torch.deg2rad(doa[sample_index]).to(self.device)
            i_values = torch.arange(self.args.n, dtype=torch.float32, device=self.device)

            angles = torch.sin(s_values)
            x = -1j * torch.pi * torch.outer(i_values, angles)

            STM_A[sample_index] = torch.exp(x)

        return STM_A

    def nonco_signal_generator(self):
        """Generates non-coherent source signals."""
        
        # Generate real and imaginary parts using torch.randn
        real_part = torch.randn(self.args.sample, self.args.l, self.args.k, 1, dtype=torch.float32, device=self.device)
        imag_part = torch.randn(self.args.sample, self.args.l, self.args.k, 1, dtype=torch.float32, device=self.device)
        
        # Scale by the factor derived from variance
        scale_factor = torch.sqrt(torch.tensor(self.args.x_var / 2, dtype=torch.float32))
        
        x = scale_factor * (real_part + 1j * imag_part)

        x = x + self.args.mean_c
        
        return x


    def generate_experiment_data(self):
        """Experiment data generation."""
        x_dire = self.doa_generator()
        
        STM_A = self.steeringmatrix(x_dire)

        x_true = self.nonco_signal_generator()
        
        y_train = torch.zeros(self.args.sample, self.args.l, self.args.n, 1, dtype=torch.cfloat, device=self.device)
        
        for j in range(self.args.sample):
            for t in range(self.args.l):
                er1 = torch.normal(mean=0.0, std=torch.sqrt(self.r2 / 2), size=(self.args.n,)).to(self.device)
                er2 = torch.normal(mean=0.0, std=torch.sqrt(self.r2 / 2), size=(self.args.n,)).to(self.device)

                y_train[j, t, :, 0] = STM_A[j].matmul(x_true[j, t, :, 0]) + er1 + er2 * 1j
                
        return x_dire, x_true, y_train


