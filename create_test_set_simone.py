# This script creates the dataset (specifically, both train and test) of 24KhZ audio from the 48KhZ one.
# The second half of the script makes the check that the KhZs are actually 24 and 48.
# If you want to hear the difference between the two versions, use the jupyter notebook: listen_audios.ipynb

import os
import librosa
import soundfile as sf


def resample_audio(input_dir, output_dir, target_sr=24000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                input_filepath = os.path.join(root, file)
                output_filepath = os.path.join(output_dir, file)
                
                print(f'Resampling {input_filepath} to {target_sr}Hz')
                
                # Carica l'audio
                y, sr = librosa.load(input_filepath, sr=None)
                
                # Esegui il resampling
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                
                # Salva l'audio resampled
                sf.write(output_filepath, y_resampled, target_sr)
                print(f'Saved resampled file to {output_filepath}')

input_directory_train = '/media/nvme_4tb/simone_data/VoiceBank/clean_trainset_28spk_wav'
output_directory_train = '/media/nvme_4tb/simone_data/VoiceBank/clean_trainset_28spk_wav_24khz'

input_directory_test = '/media/nvme_4tb/simone_data/VoiceBank/clean_testset_wav'
output_directory_test = '/media/nvme_4tb/simone_data/VoiceBank/clean_testset_wav_24khz'

resample_audio(input_directory_train, output_directory_train)
print('\n\n\n\n\n\n\n')
resample_audio(input_directory_test, output_directory_test)


#### Check ###
# Ora controlliamo che i file siano effettivamente in risoluzioni diverse:
import wave

def get_sample_rate(filename):
    with wave.open(filename, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
    return sample_rate

# Esempio di utilizzo
filename_48 = '/media/nvme_4tb/simone_data/VoiceBank/clean_testset_wav/p257_434.wav' # ho provato anche col trainset, è tutto ok
sample_rate_48 = get_sample_rate(filename_48)
print(f"La frequenza di campionamento per il file {filename_48} è: {sample_rate_48} Hz"); print('')

filename_24 = '/media/nvme_4tb/simone_data/VoiceBank/clean_testset_wav_24khz/p257_434.wav' # ho provato anche col trainset, è tutto ok
sample_rate_24 = get_sample_rate(filename_24)
print(f"La frequenza di campionamento per il file {filename_24} è: {sample_rate_24} Hz")




########## Creation of validation_set ##########
import os
import shutil
import numpy as np

def create_validation_set(input_dir, target_dir, val_split=0.2):
    '''Prende le due cartelle di x e y (24KhZ e 48KhZ) e le divide in 80/20 per creare il validation di entrambe.'''
    # Crea le directory per il validation set
    val_input_dir = input_dir + "_validation"
    val_target_dir = target_dir + "_validation"
    
    os.makedirs(val_input_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    # Lista dei file e ordinamento per garantire coerenza
    filenames = sorted(os.listdir(input_dir))
    
    # Calcola quanti file devono andare nel validation set
    val_count = int(np.floor(val_split * len(filenames)))

    # Indici per train e validation
    train_filenames = filenames[:-val_count]
    val_filenames = filenames[-val_count:]

    # Sposta i file di validation nelle nuove directory
    for filename in val_filenames:
        # Sposta i file per input e target
        shutil.move(os.path.join(input_dir, filename), os.path.join(val_input_dir, filename))
        shutil.move(os.path.join(target_dir, filename), os.path.join(val_target_dir, filename))
    
    print(f"Moved {val_count} files to validation set.")

# Definisci le cartelle originali
input_dir = '/media/nvme_4tb/simone_data/VoiceBank/clean_trainset_28spk_wav_24khz'
target_dir = '/media/nvme_4tb/simone_data/VoiceBank/clean_trainset_28spk_wav'

# Esegui la funzione per creare il validation set
create_validation_set(input_dir, target_dir, val_split=0.2)
