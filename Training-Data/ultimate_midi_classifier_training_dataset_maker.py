# -*- coding: utf-8 -*-
"""Ultimate_MIDI_Classifier_Training_Dataset_Maker.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QS-eXtY7nLUkNvacTrm47dpGU93O0QRx

# Ultimate MIDI Classifier Training Dataset Maker (ver. 1.0)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

#### Project Los Angeles

#### Tegridy Code 2024

***

# (SETUP ENVIRONMENT)
"""

#@title Install all dependencies (run only once per session)

!git clone --depth 1 https://github.com/asigalov61/tegridy-tools

#@title Import all needed modules

print('Loading needed modules. Please wait...')
import os
import copy
import math
import statistics
import random

from joblib import Parallel, delayed, parallel_config

from tqdm import tqdm

if not os.path.exists('/content/Dataset'):
    os.makedirs('/content/Dataset')

print('Loading TMIDIX module...')
os.chdir('/content/tegridy-tools/tegridy-tools')

import TMIDIX

from huggingface_hub import hf_hub_download

print('Done!')

os.chdir('/content/')
print('Enjoy! :)')

"""# (DOWNLOAD CLASSIFICATION MIDI DATASET)"""

# Commented out IPython magic to ensure Python compatibility.
#@title Download and unzip sample classification MIDI Dataset

hf_hub_download(repo_id='asigalov61/Annotated-MIDI-Dataset',
                filename='Annotated-MIDI-Dataset-Large-Raw-Version-CC-BY-NC-SA.zip',
                local_dir='/content/Dataset',
                repo_type='dataset'
                )


# %cd /content/Dataset/

!unzip Annotated-MIDI-Dataset-Large-Raw-Version-CC-BY-NC-SA.zip
!rm Annotated-MIDI-Dataset-Large-Raw-Version-CC-BY-NC-SA.zip

# %cd /content/

"""# (FILE LIST)"""

#@title Save file list

###########

print('=' * 70)
print('Loading MIDI files...')
print('This may take a while on a large dataset in particular.')

dataset_addr = "/content/Dataset"

filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames if file.endswith('.mid')]
print('=' * 70)

if filez == []:
    print('Could not find any MIDI files. Please check Dataset dir...')
    print('=' * 70)

print('Randomizing file list...')
random.shuffle(filez)

DIVIDER = math.ceil(math.sqrt(len(filez)))

print('=' * 70)

print('Creating sorted files list...')

f_names = sorted([os.path.basename(f).split('.mid')[0].split(' --- ') for f in filez], key=lambda x: (x[1], x[0]))

file_names = []

for f in f_names:
    file_names.append(' --- '.join(f))

print('Done!')
print('=' * 70)

TMIDIX.Tegridy_Any_Pickle_File_Writer([filez, file_names, DIVIDER], '/content/files_labels_divider_data')
print('=' * 70)

print('Found', len(filez), 'MIDIs')
print('=' * 70)

#@title Load file list
filez, file_names, DIVIDER = TMIDIX.Tegridy_Any_Pickle_File_Reader('/content/files_labels_divider_data')

"""# (LOAD TMIDIX MIDI PROCESSOR)"""

# @title Load TMIDIX MIDI Processor

def file_name_to_file_name_tokens(file_name):
    idx = file_names.index(file_name)

    tok1 = idx // DIVIDER
    tok2 = idx % DIVIDER

    return [tok1, tok2]

def file_name_tokens_to_file_name(file_name_tokens):

    tok1 = file_name_tokens[0]
    tok2 = file_name_tokens[1]

    idx = (tok1 * DIVIDER) + tok2

    return file_names[idx]

def TMIDIX_MIDI_Processor(midi_file):

    try:

        fn = os.path.basename(midi_file)
        fn1 = fn.split('.mid')[0]

        fn_tokens = file_name_to_file_name_tokens(fn1)

        #=======================================================
        # START PROCESSING

        #===============================================================================
        # Raw single-track ms score

        raw_score = TMIDIX.midi2single_track_ms_score(midi_file)

        #===============================================================================
        # Enhanced score notes

        escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]

        if len(escore_notes) > 0:

            #=======================================================
            # PRE-PROCESSING

            #===============================================================================
            # Augmented enhanced score notes

            escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, timings_divider=32)

            escore_notes = [e for e in escore_notes if e[6] < 80 or e[6] == 128]

            #=======================================================
            # Augmentation

            #=======================================================
            # FINAL PROCESSING

            melody_chords = []

            #=======================================================
            # MAIN PROCESSING CYCLE
            #=======================================================

            pe = escore_notes[0]

            pitches = []

            for e in escore_notes:

                #=======================================================
                # Timings...

                delta_time = max(0, min(127, e[1]-pe[1]))

                if delta_time != 0:
                    pitches = []

                # Durations and channels

                dur = max(1, min(127, e[2]))

                # Patches
                pat = max(0, min(128, e[6]))

                # Pitches

                if pat == 128:
                    ptc = max(1, min(127, e[4]))+128
                else:
                    ptc = max(1, min(127, e[4]))

                #=======================================================
                # FINAL NOTE SEQ

                # Writing final note synchronously

                if ptc not in pitches:
                    melody_chords.extend([delta_time, dur+128, ptc+256])
                    pitches.append(ptc)

                pe = e

                #=======================================================

            #=======================================================

            # TOTAL DICTIONARY SIZE 512
            #=======================================================

            return [fn_tokens, melody_chords]

        else:
            return None

    except Exception as e:
        print('=' * 70)
        print(midi_file)
        print(e)
        print('=' * 70)
        return None

"""# (PROCESS)"""

#@title Process MIDIs with TMIDIX MIDI processor

NUMBER_OF_PARALLEL_JOBS = 16 # Number of parallel jobs
NUMBER_OF_FILES_PER_ITERATION = 16 # Number of files to queue for each parallel iteration
SAVE_EVERY_NUMBER_OF_ITERATIONS = 160 # Save every 2560 files

print('=' * 70)
print('TMIDIX MIDI Processor')
print('=' * 70)
print('Starting up...')
print('=' * 70)

###########

melody_chords_f = []

files_count = 0

print('Processing MIDI files. Please wait...')
print('=' * 70)

for i in tqdm(range(0, len(filez), NUMBER_OF_FILES_PER_ITERATION)):

  with parallel_config(backend='threading', n_jobs=NUMBER_OF_PARALLEL_JOBS, verbose = 0):

    output = Parallel(n_jobs=NUMBER_OF_PARALLEL_JOBS, verbose=0)(delayed(TMIDIX_MIDI_Processor)(f) for f in filez[i:i+NUMBER_OF_FILES_PER_ITERATION])

    for o in output:

        if o is not None:
            melody_chords_f.append(o)

    files_count += len(melody_chords_f)

    # Saving every 2560 processed files
    if i % (NUMBER_OF_FILES_PER_ITERATION * SAVE_EVERY_NUMBER_OF_ITERATIONS) == 0 and i != 0:
        print('SAVING !!!')
        print('=' * 70)
        print('Saving processed files...')
        print('=' * 70)
        print('Processed so far:', files_count, 'out of', len(filez), '===', files_count / len(filez), 'good files ratio')
        print('=' * 70)
        count = str(files_count)
        TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, '/content/MIDI_CLS_INTs_'+count)
        melody_chords_f = []

        print('=' * 70)

print('SAVING !!!')
print('=' * 70)
print('Saving processed files...')
print('=' * 70)
print('Processed so far:', files_count, 'out of', len(filez), '===', files_count / len(filez), 'good files ratio')
print('=' * 70)
count = str(files_count)
TMIDIX.Tegridy_Any_Pickle_File_Writer(melody_chords_f, '/content/MIDI_CLS_INTs_'+count)
print('=' * 70)

"""# (TEST INTS)"""

#@title Test INTs

train_data1 = random.choice(melody_chords_f)

print('=' * 70)
print('Song-Artist:', file_name_tokens_to_file_name(train_data1[0]))
print('=' * 70)
print('Sample INTs', train_data1[1][:15])
print('=' * 70)

out = train_data1[1]

if len(out) != 0:

    song = out
    song_f = []

    time = 0
    dur = 0
    vel = 90
    pitch = 0
    channel = 0

    for ss in song:

        if 0 <= ss < 128:

            time += ss

        if 128 < ss < 256:

            dur = (ss-128)

        if 256 < ss < 512:

            chan = (ss-256) // 128

            if chan == 1:
                channel = 9
            else:
                channel = 0

            pitch = (ss-256) % 128

            song_f.append(['note', time, dur, channel, pitch, vel ])

detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Ultimate MIDI Classifier',
                                                        output_file_name = '/content/Ultimate-MIDI-Classifier-Composition',
                                                        track_name='Project Los Angeles',
                                                        timings_multiplier=32
                                                        )

print('=' * 70)

"""# Congrats! You did it! :)"""