# https://huggingface.co/spaces/asigalov61/Ultimate-MIDI-Classifier

import os
import time as reqtime
import datetime
from pytz import timezone

import torch

import spaces
import gradio as gr

from x_transformer_1_23_2 import *

import random
import re
from statistics import mode

import TMIDIX
    
# =================================================================================================

@spaces.GPU
def ClassifyMIDI(input_midi, input_sampling_resolution):

    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = reqtime.time()

    print('=' * 70)

    fn = os.path.basename(input_midi.name)
    fn1 = fn.split('.')[0]

    print('=' * 70)
    print('Ultimate MIDI Classifier')
    print('=' * 70)
    
    print('Input MIDI file name:', fn)

    print('=' * 70)
    print('Loading MIDI file...')
    
    midi_name = fn
    
    raw_score = TMIDIX.midi2single_track_ms_score(open(input_midi.name, 'rb').read())
    
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    
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

    notes_counter = 0

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
            notes_counter += 1

        pe = e

    #==============================================================
    
    print('Done!')
    print('=' * 70)
    
    print('Sampling score...') 
    
    chunk_size = 1020
    
    score = melody_chords
    
    input_data = []
    
    for i in range(0, len(score)-chunk_size, chunk_size // input_sampling_resolution):
        schunk = score[i:i+chunk_size]
    
        if len(schunk) == chunk_size:
    
            td = [937]
    
            td.extend(schunk)
    
            td.extend([938])
    
            input_data.append(td)
            
    print('Done!')
    print('=' * 70)
    
    #==============================================================

    classification_summary_string = '=' * 70
    classification_summary_string += '\n'

    samples_overlap = 340 - chunk_size // input_sampling_resolution // 3
    
    print('Composition has', notes_counter, 'notes')
    print('=' * 70)
    print('Composition was split into' , len(input_data), 'samples', 'of 340 notes each with', samples_overlap, 'notes overlap')
    print('=' * 70)
    print('Number of notes in all composition samples:', len(input_data) * 340)
    print('=' * 70)

    classification_summary_string += 'Composition has ' + str(notes_counter) + ' notes\n'
    classification_summary_string += '=' * 70
    classification_summary_string += '\n'
    classification_summary_string += 'Composition was split into ' + 'samples of 340 notes each with ' + str(samples_overlap) + ' notes overlap\n'
    classification_summary_string += 'Number of notes in all composition samples: ' + str(len(input_data) * 340) + '\n'
    classification_summary_string += '=' * 70
    classification_summary_string += '\n'

    print('Loading model...')
    
    SEQ_LEN = 1026
    PAD_IDX = 940
    DEVICE = 'cuda' # 'cuda'
    
    # instantiate the model
    
    model = TransformerWrapper(
        num_tokens = PAD_IDX+1,
        max_seq_len = SEQ_LEN,
        attn_layers = Decoder(dim = 1024, depth = 24, heads = 32, attn_flash = True)
    )
    
    model = AutoregressiveWrapper(model, ignore_index=PAD_IDX, pad_value=PAD_IDX)
    
    model = torch.nn.DataParallel(model)
    
    model.to(DEVICE)
    
    print('=' * 70)
    
    print('Loading model checkpoint...')
    
    model.load_state_dict(
        torch.load('Ultimate_MIDI_Classifier_Trained_Model_29886_steps_0.556_loss_0.8339_acc.pth',
                   map_location=DEVICE))
    print('=' * 70)
    
    if DEVICE == 'cpu':
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16
    
    ctx = torch.amp.autocast(device_type=DEVICE, dtype=dtype)

    print('Done!')
    print('=' * 70)

    #==================================================================

    print('=' * 70)
    print('Ultimate MIDI Classifier')
    print('=' * 70)
    print('Classifying...')
    
    torch.cuda.empty_cache()
    
    model.eval()

    artist_results = []
    song_results = []
    
    results = []
    
    for input in input_data:
    
        x = torch.tensor(input[:1022], dtype=torch.long, device='cuda')
    
        with ctx:
          out = model.module.generate(x,
                                      2,
                                      filter_logits_fn=top_k,
                                      filter_kwargs={'k': 1},
                                      temperature=0.9,
                                      return_prime=False,
                                      verbose=False)
    
        result = tuple(out[0].tolist())
    
        results.append(result)
    
    final_result = mode(results)
    
    print('=' * 70)
    print('Done!')
    print('=' * 70)
    
    result_toks = [final_result[0]-512, final_result[1]-512]
    mc_song_artist = song_artist_tokens_to_song_artist(result_toks)
    gidx = genre_labels_fnames.index(mc_song_artist)
    mc_genre = genre_labels[gidx][1]
    
    print('Most common classification genre label:', mc_genre)
    print('Most common classification song-artist label:', mc_song_artist)
    print('Most common song-artist classification label ratio:' , results.count(final_result) / len(results))
    print('=' * 70)
    
    classification_summary_string += 'Most common classification genre label: ' + str(mc_genre) + '\n'
    classification_summary_string += 'Most common classification song-artist label: ' + str(mc_song_artist) + '\n'
    classification_summary_string += 'Most common song-artist classification label ratio: '+ str(results.count(final_result) / len(results)) + '\n'
    classification_summary_string += '=' * 70
    classification_summary_string += '\n' 
    
    print('All classification labels summary:')
    print('=' * 70)
    
    all_artists_labels = []
    
    for i, res in enumerate(results):
        result_toks = [res[0]-512, res[1]-512]
        song_artist = song_artist_tokens_to_song_artist(result_toks)
        gidx = genre_labels_fnames.index(song_artist)
        genre = genre_labels[gidx][1]
        
        print('Notes', i*(340-samples_overlap), '-', (i*(340-samples_overlap))+340, '===', genre, '---', song_artist)
        classification_summary_string += 'Notes ' + str(i*samples_overlap) + ' - ' +  str((i*samples_overlap)+340) + ' === ' + str(genre) + ' --- ' + str(song_artist) + '\n'
        
        artist_label = str_strip_artist(song_artist.split(' --- ')[1])
        
        all_artists_labels.append(artist_label)
        
    classification_summary_string += '=' * 70
    classification_summary_string += '\n'    
    print('=' * 70)
    
    mode_artist_label = mode(all_artists_labels)
    mode_artist_label_count = all_artists_labels.count(mode_artist_label)
    
    print('Aggregated artist classification label:', mode_artist_label)
    print('Aggregated artist classification label ratio:', mode_artist_label_count / len(all_artists_labels))
    classification_summary_string += 'Aggregated artist classification label: ' + str(mode_artist_label) + '\n'
    classification_summary_string += 'Aggregated artist classification label ratio: ' + str(mode_artist_label_count / len(all_artists_labels)) + '\n'
    classification_summary_string += '=' * 70
    classification_summary_string += '\n'
    
    print('=' * 70)
    print('Done!')
    print('=' * 70)
    
    #========================================================
    
    print('-' * 70)
    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('-' * 70)
    print('Req execution time:', (reqtime.time() - start_time), 'sec')

    return classification_summary_string

# =================================================================================================

if __name__ == "__main__":
    
    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)

    #===============================================================================
    # Helper functions
    #===============================================================================
    
    def str_strip_song(string):
      if string is not None:
        string = string.replace('-', ' ').replace('_', ' ').replace('=', ' ')
        str1 = re.compile('[^a-zA-Z ]').sub('', string)
        return re.sub(' +', ' ', str1).strip().title()
      else:
        return ''
    
    def str_strip_artist(string):
      if string is not None:
        string = string.replace('-', ' ').replace('_', ' ').replace('=', ' ')
        str1 = re.compile('[^0-9a-zA-Z ]').sub('', string)
        return re.sub(' +', ' ', str1).strip().title()
      else:
        return ''
    
    def song_artist_to_song_artist_tokens(file_name):
        idx = classifier_labels.index(file_name)
    
        tok1 = idx // 424
        tok2 = idx % 424
    
        return [tok1, tok2]
    
    def song_artist_tokens_to_song_artist(file_name_tokens):
    
        tok1 = file_name_tokens[0]
        tok2 = file_name_tokens[1]
    
        idx = (tok1 * 424) + tok2
    
        return classifier_labels[idx]
    
    #===============================================================================
    
    print('=' * 70)
    print('Loading Ultimate MIDI Classifier labels...')
    print('=' * 70)
    classifier_labels = TMIDIX.Tegridy_Any_Pickle_File_Reader('Ultimate_MIDI_Classifier_Song_Artist_Labels')
    print('=' * 70)
    genre_labels = TMIDIX.Tegridy_Any_Pickle_File_Reader('Ultimate_MIDI_Classifier_Music_Genre_Labels')
    genre_labels_fnames = [f[0] for f in genre_labels]
    print('=' * 70)
    print('Done!')
    print('=' * 70)

    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Ultimate MIDI Classifier</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Classify absolutely any MIDI by genre, song and artist</h1>")
        gr.Markdown(
            "![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Ultimate-MIDI-Classifier&style=flat)\n\n"
            "This is a demo for Ultimate MIDI Classifier\n\n"
            "Check out [Ultimate MIDI Classifier](https://github.com/asigalov61/Ultimate-MIDI-Classifier) on GitHub!\n\n"
            "[Open In Colab]"
            "(https://colab.research.google.com/github/asigalov61/Ultimate-MIDI-Classifier/blob/main/Ultimate_MIDI_Classifier.ipynb)"
            " for all options, faster execution and endless classification"
        )

        gr.Markdown("## Upload any MIDI to classify")
        
        input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"])
        input_sampling_resolution = gr.Slider(1, 5, value=2, step=1, label="Classification sampling resolution")

        run_btn = gr.Button("classify", variant="primary")

        gr.Markdown("## Classification results")

        output_midi_cls_summary = gr.Textbox(label="MIDI classification results")

        run_event = run_btn.click(ClassifyMIDI, [input_midi, input_sampling_resolution],
                                  [output_midi_cls_summary])
        gr.Examples(
            [["Honesty.kar", 2], 
             ["House Of The Rising Sun.mid", 2], 
             ["Nothing Else Matters.kar", 2],
             ["Sharing The Night Together.kar", 2]
            ],
            [input_midi, input_sampling_resolution],
            [output_midi_cls_summary],
            ClassifyMIDI,
            cache_examples=True,
        )
        
        app.queue().launch()