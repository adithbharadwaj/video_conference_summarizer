import speech_recognition as sr

import os

from pydub import AudioSegment
from pydub.silence import split_on_silence

from pydub import AudioSegment
import speech_recognition as sr

from scipy.io import wavfile
import os
import numpy as np
import argparse
from tqdm import tqdm

def convert_audio_to_text(path = "gandhi.wav"):
        # Input audio file to be sliced
        from pydub import AudioSegment
        #path = 'audio.wav'
        print(path)
        #audio = AudioSegment.from_mp3(path)
        #audio.export("gsk_mom.wav", format="wav")
        audio = AudioSegment.from_wav(path)

        # Length of the audiofile in milliseconds
        n = len(audio)
        counter = 1
        fh = open("recognized.txt", "w+")

        try:
                os.mkdir('audio_chunks')
        except(FileExistsError):
                pass

        os.chdir('audio_chunks')

        # Interval length at which to slice the audio file.
        interval = 10 * 1000
        overlap = 1*1000
        start = 0
        end = 0

        # Flag to keep track of end of file.
        flag = 0
        # Iterate from 0 to end of the file,
        for i in range(0, 2 * n, interval):

            # During first iteration, start is 0, end is the interval
            if i == 0:
                start = 0
                end = interval

            # All other iterations, start is the previous end - overlap end becomes end + interval
            else:
                start = end - overlap
                end = start + interval

            # When end becomes greater than the file length, end is set to the file length
            # flag is set to 1 to indicate break.
            if end >= n:
                end = n
                flag = 1

            # Storing audio file from the defined start to end
            chunk = audio[start:end]

            filename = 'chunk'+str(counter)+'.wav'
            chunk.export(filename, format ="wav")
            print("Processing chunk "+str(counter)+". Start = "
                                +str(start//1000)+"s end = "+str(end//1000))

            counter = counter + 1

            AUDIO_FILE = filename
            r = sr.Recognizer()

            with sr.AudioFile(AUDIO_FILE) as source:
                #r.adjust_for_ambient_noise(source)
                audio_listened = r.listen(source)

            try:
                rec = r.recognize_google(audio_listened)
                fh.write(rec+". ")

            except sr.UnknownValueError:
                print("Could not understand audio")

            except sr.RequestError as e:
                print("Could not request results. check your internet connection")

            # Check for flag. If flag is 1, end of the whole audio reached.
            if flag == 1:
                fh.close()
                break

        os.chdir('..')

def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def silence_based_conversion(path = "alice-medium.wav"):
        #print(path)
        #sound = AudioSegment.from_mp3(path)
        #sound.export('audio.wav', format="wav")
        song = AudioSegment.from_wav(path)
        fh = open("recognized.txt", "w+")
        #split track where silence is 2 seconds or more and get chunks

        chunks = split_on_silence(song,
            # must be silent for at least 2 seconds or 2000 ms
            min_silence_len=1800,

            # consider it silent if quieter than -16 dBFS
            #Adjust this per requirement
            silence_thresh = -22
        )

        try:
                os.mkdir('audio_chunks')
        except(FileExistsError):
                pass

        os.chdir('audio_chunks')

        start = 0
        end = 0

        #Process each chunk per requirements
        for i, chunk in enumerate(chunks):
            #Create 0.5 seconds silence chunk
            silence_chunk = AudioSegment.silent(duration=10)

            #Add  0.5 sec silence to beginning and end of audio chunk
            audio_chunk = silence_chunk + chunk + silence_chunk

            #Normalize each audio chunk
            normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

            #Export audio chunk with new bitrate
            print("exporting chunk{0}.wav".format(i))
            normalized_chunk.export("./chunk{0}.wav".format(i), bitrate='192k', format="wav")

            filename = 'chunk'+str(i)+'.wav'
            #chunk.export(filename, format ="wav")
            print("Processing chunk "+str()+". Start = "
                                +str(start//1000)+"s end = "+str(end//1000))

            AUDIO_FILE = filename
            r = sr.Recognizer()

            with sr.AudioFile(AUDIO_FILE) as source:
                #r.adjust_for_ambient_noise(source)
                audio_listened = r.listen(source)

            try:
                rec = r.recognize_google(audio_listened)
                fh.write(rec+". ")

            except sr.UnknownValueError:
                print("Could not understand audio")

            except sr.RequestError as e:
                print("Could not request results. check your internet connection")

        os.chdir('..')

# Utility functions

def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size - 1
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]

def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))

def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1


def optimal_split(path = "mohandas.wav"):

	fh = open("recognized.txt", "w+")

	input_filename = path#args.input_file
	window_duration = 1#args.min_silence_length
	step_duration = window_duration / 10.
	silence_threshold = 0.00001#args.silence_threshold
	output_dir = './audio_chunks'
	output_filename_prefix = ''
	dry_run = False

	print("Splitting {} where energy is below {}% for longer than {}s.".format(
	    input_filename,
	    silence_threshold * 100.,
	    window_duration
	))

	try:
		os.mkdir('audio_chunks')
	except(FileExistsError):
		pass

	# Read and split the file

	sample_rate, samples = input_data=wavfile.read(filename=input_filename, mmap=True)

	max_amplitude = np.iinfo(samples.dtype).max
	max_energy = energy([max_amplitude])

	window_size = int(window_duration * sample_rate)
	step_size = int(step_duration * sample_rate)

	signal_windows = windows(
	    signal=samples,
	    window_size=window_size,
	    step_size=step_size
	)

	window_energy = (energy(w) / max_energy for w in tqdm(
	    signal_windows,
	    total=int(len(samples) / float(step_size))
	))

	window_silence = (e > silence_threshold for e in window_energy)

	cut_times = (r * step_duration for r in rising_edges(window_silence))

	# This is the step that takes long, since we force the generators to run.
	print("Finding silences...")
	cut_samples = [int(t * sample_rate) for t in cut_times]
	cut_samples.append(-1)
	cut_ranges = [(i, cut_samples[i], cut_samples[i+1]) for i in range(len(cut_samples) - 1)]
	os.chdir('audio_chunks')
	for i, start, stop in tqdm(cut_ranges):
	    output_file_path = "chunk{}.wav".format(i)
	    if not dry_run:
	        print("Writing file {}".format(output_file_path))
	        wavfile.write(filename = output_file_path, rate = sample_rate, data = samples[start:stop])
	        aud = AudioSegment.from_wav(output_file_path)
	        aud = aud[1:]
	        aud.export(output_file_path, format = "wav")
	        

        	AUDIO_FILE = output_file_path
        	r = sr.Recognizer()
        	with sr.AudioFile(AUDIO_FILE) as source:
            	#r.adjust_for_ambient_noise(source)
        		audio_listened = r.listen(source)

        	try:
        		rec = r.recognize_google(audio_listened)
        		fh.write(rec+". ")

        	except sr.UnknownValueError:
        		print("Could not understand audio")

        	except sr.RequestError as e:
        		print("Could not request results. check your internet connection")

	else:
	    print("Not writing file {}".format(output_file_path))

	os.chdir('..')