def random_power(images, power, c):
    images = images - images.min()
    images = images/(images.max()+0.0000001)
    images = images**(random.random()*power + c)
    return images    
    
    # Second and third class addition
    for idx, file in enumerate(batch_x):
      
      file_1 = data_x[idx]
      y_entry = np.zeros(397)
      file_1_label = file.split('_')[0]
      file_1_label_idx = CLASS_TO_INT[file_1_label]
      
      
      indices_same = True
      while indices_same:
        idx2 = random.randint(0, len(self.filenames) - 1) # Second file
        idx3 = random.randint(0, len(self.filenames) - 1) # Third file
        
        indices_same = (self.filenames[idx2] == self.filenames[idx3] or file == self.filenames[idx2] or file == self.filenames[idx3])
          
        
      file_2 = np.load(self.path + self.filenames[idx2])
      file_2 = np.roll(file_2, random.randint(INPUT_SHAPE[1] / 16, INPUT_SHAPE[1] / 2),axis=1)
      file_2_label = self.filenames[idx2].split('_')[0]
      file_2_label_idx = CLASS_TO_INT[file_2_label]
      
      file_3 = np.load(self.path + self.filenames[idx3])
      file_3 = np.roll(file_3, random.randint(INPUT_SHAPE[1] / 16, INPUT_SHAPE[1] / 2),axis=1)
      file_3_label = self.filenames[idx3].split('_')[0]
      file_3_label_idx = CLASS_TO_INT[file_3_label]
      
      
      # file_1_max, file_2_max, file_3_max = file_1.max(), file_2.max(), file_3.max()
      # file_max_idx = np.argmax([file_1_max, file_2_max, file_3_max])
      
      # if file_max_idx == 0:
      #   y_1 = 1
      #   y_2 = 1 - (file_1_max - file_2_max)/(file_1_max + file_2_max)
      #   y_3 = 1 - (file_1_max - file_3_max)/(file_1_max + file_3_max)
        
      # elif file_max_idx == 1:
      #   y_1 = 1 - (file_2_max - file_1_max)/(file_2_max + file_1_max)
      #   y_2 = 1
      #   y_3 = 1 - (file_2_max - file_3_max)/(file_2_max + file_3_max)
        
      # elif file_max_idx == 2:
      #   y_1 = 1 - (file_3_max - file_1_max)/(file_3_max + file_1_max)
      #   y_2 = 1 - (file_3_max - file_2_max)/(file_3_max + file_2_max)
      #   y_3 = 1
      
      # NORMALIZE
      file_1 **= random_power
      
      r2 = random.random()
      r3 = random.random()
      
      if r2 < 0.7 and r3 > 0.35:    # 45.5% 2 classes
        # NORMALIZE
        file_2 **= random_power
        
        # if y_3 == 1:
        #   if file_1_max > file_2_max:
        #     y_1 = 1
        #     y_2 = 1 - (file_1_max - file_2_max)/(file_1_max + file_2_max)
        #   else:
        #     y_1 = 1 - (file_2_max - file_1_max)/(file_2_max + file_1_max)
        #     y_2 = 1
          
        y_entry[file_1_label_idx] = 1
        y_entry[file_2_label_idx] = 1
        
        file_1 += file_2
        
      elif r2 < 0.7 and r3 < 0.35:    # 24.5% 3 classes
        # NORMALIZE
        file_2 **= random_power
        
        # NORMALIZE
        file_3 **= random_power
        
        y_entry[file_1_label_idx] = 1
        
        y_entry[file_2_label_idx] = 1
        file_1 += file_2
        
        y_entry[file_3_label_idx] = 1
        file_1 += file_3
        
      else:
        y_entry[file_1_label_idx] = 1
      
      data_x[idx] = file_1
      batch_y[idx] = y_entry

    # TO DECIBELS
    for i in range(self.batch_size):  
      data_x[i] = librosa.power_to_db(data_x[i], ref=np.max)
      data_x[i] = (data_x[i] + 80) / 80 
    
    # Add white noise
    if random.random()<0.8:
      for i in range(self.batch_size):
        white_noise = (np.random.sample((INPUT_SHAPE[0], INPUT_SHAPE[1])).astype(np.float32) + 9) * data_x[i].mean() * NOISE_LEVEL * (np.random.sample() + 0.3)
        data_x[i] = data_x[i] + white_noise
    
    # Add bandpass noise
    if random.random()<0.7:
      for i in range(self.batch_size):
        a = random.randint(0, INPUT_SHAPE[0]//2)
        b = random.randint(a + 20, INPUT_SHAPE[0])
        data_x[i, a:b, :] += (np.random.sample((b - a, INPUT_SHAPE[1])).astype(np.float32) + 9) * 0.05 * data_x[i].mean() * NOISE_LEVEL  * (np.random.sample() + 0.3)
    
    
    # NORMALIZE
    for i in range(self.batch_size):
      data_x[i] = data_x[i] - data_x[i].min()
      data_x[i] = data_x[i]/data_x[i].max()
    
    # Add 3 channels
    rgb_batch = np.repeat(data_x[..., np.newaxis], 3, -1)
