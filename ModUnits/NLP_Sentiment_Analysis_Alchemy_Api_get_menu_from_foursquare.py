# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:37:57 2015

@author: dwoo57
"""

#!/usr/bin/env python

#	Copyright 2013 AlchemyAPI
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from __future__ import print_function
import re

import requests
import nltk

try:
    from urllib.request import urlopen
    from urllib.parse import urlparse
    from urllib.parse import urlencode
except ImportError:
    from urlparse import urlparse
    from urllib2 import urlopen
    from urllib import urlencode

try:
    import json
except ImportError:
    # Older versions of Python (i.e. 2.4) require simplejson instead of json
    import simplejson as json

from alchemyapi import AlchemyAPI


def GetTextFromUrl(url):
    
    demo_url = url
    
    response = alchemyapi.text('url', demo_url,{'useMetadata':0})
    print('Processing url: ', demo_url)
    
    if response['status'] == 'OK':
        print('## Response Object ##')
        print(json.dumps(response, indent=4))
    
        print('')
        print('## Keywords ##')
        #print('text: ', response['text'].encode('utf-8'))
        #for keyword in response['keywords']:
            #if keyword['sentiment']['type'] == 'positive':
                #print('text: ', keyword['text'].encode('utf-8'))
                #print('relevance: ', keyword['relevance'])
                #print('sentiment: ', keyword['sentiment']['type'])
                #if 'score' in keyword['sentiment']:
                #    print('sentiment score: ' + keyword['sentiment']['score'])
                #print('')
    else:
        print('Error in keyword extaction call: ', response['statusInfo'])

    #1.first parse break up one new line        
    test_text = response['text']
    #test_text = test_text.split("\n")  
    
    return test_text
    
def TargetedSentiment(entity_list,url):
    
    for entity in entity_list:
        
        response = alchemyapi.sentiment_targeted('url', url, entity, {'sentiment': 1})
        print('Processing url: ', url)
        
        if response['status'] == 'OK':
            #print('## Response Object ##')
            #print(json.dumps(response, indent=4))
        
            print('')
            print('## Targeted Keywords ##')
            print('text: ', entity)
            print('sentiment: ', response['docSentiment']['type'])
            #print('sentiment score: ' + response['docSentiment']['score'])
            #for keyword in response['docSentiment']:
                #if keyword['sentiment']['type'] == 'positive':
                    #print('text: ', keyword['text'].encode('utf-8'))
                    #print('relevance: ', keyword['relevance'])
                    #print('sentiment: ', keyword['sentiment']['type'])
                    #if 'score' in keyword['sentiment']:
                    #    print('sentiment score: ' + keyword['sentiment']['score'])
                    #print('')
        else:
            print('text: ', entity)
            print('Error in keyword extaction call: ', response['statusInfo'])


if __name__ == '__main__':
    """
    Writes the API key to api_key.txt file. It will create the file if it doesn't exist.
    This function is intended to be called from the Python command line using: python alchemyapi YOUR_API_KEY
    If you don't have an API key yet, register for one at: http://www.alchemyapi.com/api/register.html

    INPUT:
    argv[1] -> Your API key from AlchemyAPI. Should be 40 hex characters

    OUTPUT:
    none
    """

    import sys
    if len(sys.argv) == 2 and sys.argv[1]:
        if len(sys.argv[1]) == 40:
            # write the key to the file
            f = open('api_key.txt', 'w')
            f.write(sys.argv[1])
            f.close()
            print('Key: ' + sys.argv[1] + ' was written to api_key.txt')
            print(
                'You are now ready to start using AlchemyAPI. For an example, run: python example.py')
        else:
            print(
                'The key appears to invalid. Please make sure to use the 40 character key assigned by AlchemyAPI')
    
    #demo_url = 'http://www.yelp.com/biz/ristorante-bonaroti-vienna'
    demo_url = 'https://foursquare.com/v/xo-taste/4b788676f964a5203fd42ee3/menu' 
    
    response = alchemyapi.keywords('url', demo_url, {'sentiment': 1})
    print('Processing url: ', demo_url)
    
    if response['status'] == 'OK':
        print('## Response Object ##')
        print(json.dumps(response, indent=4))
    
        print('')
        print('## Keywords ##')
        #for keyword in response['keywords']:
            #if keyword['sentiment']['type'] == 'positive':
                #print('text: ', keyword['text'].encode('utf-8'))
                #print('relevance: ', keyword['relevance'])
                #print('sentiment: ', keyword['sentiment']['type'])
                #if 'score' in keyword['sentiment']:
                #    print('sentiment score: ' + keyword['sentiment']['score'])
                #print('')
    else:
        print('Error in keyword extaction call: ', response['statusInfo'])
        
    
    demo_url = 'https://foursquare.com/v/xo-taste/4b788676f964a5203fd42ee3/menu' 
    
    response = alchemyapi.text('url', demo_url,{'useMetadata':0})
    print('Processing url: ', demo_url)
    
    if response['status'] == 'OK':
        print('## Response Object ##')
        print(json.dumps(response, indent=4))
    
        print('')
        print('## Keywords ##')
        #print('text: ', response['text'].encode('utf-8'))
        #for keyword in response['keywords']:
            #if keyword['sentiment']['type'] == 'positive':
                #print('text: ', keyword['text'].encode('utf-8'))
                #print('relevance: ', keyword['relevance'])
                #print('sentiment: ', keyword['sentiment']['type'])
                #if 'score' in keyword['sentiment']:
                #    print('sentiment score: ' + keyword['sentiment']['score'])
                #print('')
    else:
        print('Error in keyword extaction call: ', response['statusInfo'])

    #1.first parse break up one new line        
    text = response['text']
    text = text.split("\n")   
    
    mod_text=[]
    mod_text_pos=[]    
    
    #2. only keep it when it is in english. Remove prices and menu items between 2 and 6 words
    for word in test_text:
        if word.replace(".","").isdigit() or word == "":
            continue
        else:
            if len(word.split()) <= 6 and len(word.split()) >= 2:
                mod_text.append(word)
                
                pos_test = nltk.pos_tag(nltk.word_tokenize(word))
                index = 0
                temp_pos = ""
                for pos_word in pos_test:
                    temp_pos = temp_pos + " " + pos_test[index][1]
                
                mod_text_pos.append(word)
                mod_text_pos.append(temp_pos)
    #print (mod_text)        
    #print (mod_text_pos)
    #3. Now tag words within string. looking to see what the pattern is. POS tagging.
    
    #4. Seems like most menus are made up of a combination of proper nouns. is this because of the sentence?

    # a few test    
    # this was ok all proper nouns    
    text = "So, I got the Lemograss Chicken Soup with vermicelli."
    pos_test = nltk.pos_tag(nltk.word_tokenize(text))
    
    # this has a combination
    #text ="I first started with the pork belly bun and it was very tender and flavorful..I highly recommend."
    
    #text ="Chefs Special Seafood Fried rice...used to come out light and fluffly with at least a couple pieces of Lobster"
    
    #text ="OMG you HAVE to try the walnut shrimp."
    
    # propert noun has capital letters and one of a kind word
    
    # actually using proper noun may not be a bad idea
    
    url= "http://www.yelp.com/biz/hong-kong-pearl-seafood-restaurant-falls-church"
    text= GetTextFromUrl(url)
    pos_test = nltk.pos_tag(nltk.word_tokenize(text))
    
    # this runs through and gets a list of noun phrases    
    grammar = "NP: {<NNP>*}" 
    cp = nltk.RegexpParser(grammar)
    chunked = cp.parse(pos_test)
    #print(chunked)
    
    phrase_list=[]
    
    index = 0
    for n in chunked:
        phrase =""
        if isinstance(n, nltk.tree.Tree):               
            if chunked[index].label() == 'NP':
                sub_index = 0
                for m in chunked[index]:
                    single_phrase = chunked[index][sub_index][0]
                    single_phrase = re.sub("[^a-zA-Z0-9]","",single_phrase)
                    if sub_index ==0:
                        phrase = single_phrase
                    else:
                        phrase = phrase + " " + single_phrase
                    sub_index = sub_index + 1
                
                #chunked[index][1][0]
                #print(n)
                #do_something_with_subtree(n)
            if len(phrase.split(" "))>=2: 
                phrase_list.append(phrase)
        index = index + 1
                #do_something_with_leaf(n)
        
    TargetedSentiment(phrase_list,url)
    
    #for subtree in result.subtrees():
       # if subtree.label() == 'CHUNK': print(subtree):
            
    #http://stackoverflow.com/questions/28365626/how-to-output-nltk-chunks-to-file
    #http://www.nltk.org/howto/chunk.html
    
        
    
