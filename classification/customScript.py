import os
import numpy as np
import torch
import torch.utils.data as data
import pickle as pkl
import rasterio

#SEN12MS Class - Data Generating class
class SEN12MS(data.Dataset):
    def __init__(self,cloudy_dataset = False, localComputer = False):
        
       # data path
        if localComputer:
            data_dir = "SEN12MS\data"               # SEN12MS dir
            list_dir = "SEN12MS\label_split_dir"    # split lists/ label dirs
        else:
            data_dir = "/media/dharani/SpeicherTUM/patrickebel/SEN12MS/m1474000"
            list_dir = "/home/dharani/Desktop/SEN12MS/label_split_dir"    # split lists/ label dirs
    
   
        # For now, since we are using only S2 data in local computer, these flags make no sense
        # self.use_s2 = True
        # self.use_s1=False
        # self.use_RGB = True
        
        #Deciding the no of classes for our input
        self.IGBP_s = True
        if self.IGBP_s == True:
             self.n_classes = 10
        else:
            self.n_classes = 17 
        
        #Loading the labels
        label_file_path = os.path.join(list_dir,"IGBP_probability_labels.pkl") #Local : "SEN12MS\\label_split_dir\\IGBP_probability_labels.pkl" os.path.join(list_dir,"IGBP_probability_labels.pkl")
        label_file = open(label_file_path, "rb")
        self.labels = pkl.load(label_file)
        label_file.close()
        
        #Loading the samples
        samples_file_path = os.path.join(list_dir,"train_list.pkl") #Local : "SEN12MS\\label_split_dir\\val_list.pkl" os.path.join(list_dir,"val_list.pkl")
        sample_file = open(samples_file_path, "rb")
        self.samples =  pkl.load(sample_file)
        sample_file.close()

        #Fixing the right in-depth path for all files 
        self.sampleObject = []    
        for s2_id in self.samples:
            mini_name = s2_id.split("_")
            if cloudy_dataset:
                s2_loc = os.path.join(data_dir, (mini_name[0]+'_'+mini_name[1] +'_'+ mini_name[2]+'_'+mini_name[3]),(mini_name[2]+'_'+mini_name[3] + '_'+ mini_name[4]), s2_id)
            else:
                s2_loc = os.path.join(data_dir, (mini_name[0]+'_'+ mini_name[1]), (mini_name[2]+'_'+mini_name[3]), s2_id)
            s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
            self.sampleObject.append({"s1": s1_loc, "s2": s2_loc, "id": s2_id})
       
    def __getitem__(self, index,use_all = True):
        #Bands to read with
        S2_BANDS_LD = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
        S2_BANDS_RGB = [2, 3, 4] # B(2),G(3),R(4)
        S2_BANDS_ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        with rasterio.open(self.sampleObject[index]['s2']) as data:
            if use_all:
                img = data.read(S2_BANDS_ALL)
            else:
                img = data.read(S2_BANDS_LD)
        img = img.astype(np.float32)

        # load label
        lc = self.labels[self.samples[index]]
    
        # convert label to IGBP simplified scheme
        cls1 = sum(lc[0:5]);
        cls2 = sum(lc[5:7]); 
        cls3 = sum(lc[7:9]);
        cls6 = lc[11] + lc[13];
        lc = np.asarray([cls1, cls2, cls3, lc[9], lc[10], cls6, lc[12], lc[14], lc[15], lc[16]])
        
        loc = np.argmax(lc, axis=-1)
        lc_hot = np.zeros_like(lc).astype(np.float32)
        lc_hot[loc] = 1

        rt_sample = {'image': img, 'label': lc_hot, 'id': self.samples[index]}
        return rt_sample

    def __len__(self):
        return len(self.samples)


def main():

    #Data Generation to pass to Dataloader
    #imgTransform = transforms.Compose([ToTensor(),Normalize(bands_mean, bands_std)]) # No idea what this does
    lengthArray = np.arange(0,SEN12MS().__len__(),1)
    bandArray = np.arange(0,13,1)
    for i in bandArray:  
        singleBandTotalForAllImages = np.zeros((256,256))
        for j in lengthArray:
            train_dataGen = SEN12MS().__getitem__(index = j,use_all = True)
            singleBandTotalForAllImages = singleBandTotalForAllImages + train_dataGen['image'][i]
        temp = singleBandTotalForAllImages/SEN12MS().__len__()
        bandmean = np.mean(temp)
        print(bandmean)   
        #Starting with DataLoader
        train_data_loader = data.DataLoader(train_dataGen, 
                                   batch_size=64, 
                                   num_workers=0, 
                                   shuffle=True, 
                                   pin_memory=True)
        print('Completed')


if __name__ == "__main__":
    main()
         
      
        