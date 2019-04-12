import speech_recognition as sr

import os

from pydub import AudioSegment
from pydub.silence import split_on_silence

from pydub import AudioSegment 
import speech_recognition as sr 

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
	    	r.adjust_for_ambient_noise(source)
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
	    min_silence_len=1600,

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
	    silence_chunk = AudioSegment.silent(duration=500)

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
	    	r.adjust_for_ambient_noise(source)
	    	audio_listened = r.listen(source) 
	  
	    try:     
	        rec = r.recognize_google(audio_listened) 
	        fh.write(rec+". ") 

	    except sr.UnknownValueError: 
	        print("Could not understand audio") 
	 
	    except sr.RequestError as e: 
	        print("Could not request results. check your internet connection") 

	os.chdir('..')
	  

#silence_based_conversion()
#convert_audio_to_text()