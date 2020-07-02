from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import json
import os
from pocketsphinx import DefaultConfig, Decoder, get_model_path, get_data_path
import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize

# Create your views here.

model_path = get_model_path()
data_path = get_data_path()

# Create a decoder with a certain model
config = DefaultConfig()
config.set_string('-hmm', os.path.join(model_path, 'en-us'))
config.set_string('-lm', os.path.join(model_path, 'en-us.lm.bin'))
config.set_string('-dict', os.path.join(model_path, 'cmudict-en-us.dict'))
decoder = Decoder(config)



@api_view(["POST"])
def ReadAudioFile(request):
    text = speechToText(request.FILES['audio'])
    handle_uploaded_file(request.FILES['audio'])
    responseData = Response({'result': text})
    responseData['Access-Control-Allow-Origin'] = '*'
    return responseData


def handle_uploaded_file(f):
    with open('output.wav', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def speechToText(blob):
    buf = bytearray(1024)
    decoder.start_utt()
    while blob.readinto(buf):
        decoder.process_raw(buf, False, False)
    decoder.end_utt()
    text = decoder.hyp().hypstr
    word = text.lower()
    word_tokens = word_tokenize(word)
    extracted_word=[]
    textstring = ""
    for i in word_tokens:
        for j,pos in nltk.pos_tag(nltk.word_tokenize(str(i))):
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'VB' or pos == 'CD'):
                extracted_word.append(j)
                textstring = ' '.join(extracted_word)
    return textstring







##@api_view(["POST"])
##def ReadAudioFile(request):
##    try:
####      data = handle_uploaded_audio_file(request.FILES['file'])
##        handle_uploaded_audio_file(request.FILES['audio'])
##        responseData = JsonResponse({'result':'true'})
##        responseData['Access-Control-Allow-Origin']='*'
##        return responseData
##    except ValueError as e:
##        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
##
##
##
##def handle_uploaded_file(f):
##    with open('test.wav', 'wb+') as destination:
##        for chunk in f.chunks():
##            destination.write(chunk)
##
##def handle_uploaded_audio_file(f):
##    buf = bytearray(1024)
##    with open('test.wav', 'wb+') as destination:
##        for chunk in f.chunks():
##            destination.write(chunk)
##    with open('test.wav', 'rb') as audioF:
##        decoder.start_utt()
##        while audioF.readinto(buf):
##            decoder.process_raw(buf, False, False)
##        decoder.end_utt()
##    text = decoder.hyp().hypstr
##    return text
##
##
##
##
####def handle_uploaded_audio_file(f):
####    buf = bytearray(1024)
####    decoder.start_utt()
####    while f.readinto(buf):
####        decoder.process_raw(buf, False, False)
####    decoder.end_utt()
####    text = decoder.hyp().hypstr
####    return text
####
