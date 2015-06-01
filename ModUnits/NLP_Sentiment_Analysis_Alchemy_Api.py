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

import requests

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

    review_list=[]
    review_list.append("The sticky rice roll (fan tuan) is also very good and hard to come by at Chinese restaurants so I was impressed!" )
    review_list.append("They had a delicious Singapore style steamed chicken over rice (rice is long grain cooked in chicken broth).")
    review_list.append("I usually stick to the godeunguh, which always comes out perfectly-grilled, salted perfectly, and great to eat with the bop.")
    review_list.append("Manti and Plov, and the lamb in both dishes was just so tender and tasty.")
    review_list.append("The dumplings barely had any soup in there")
    review_list.append("According to my dad (who, keep in mind, has not been back to Taiwan in 15+ years), said that the oyster vermicelli's taste was off.") 
    review_list.append("So currently this is what you get for 40 bucks here:  bland snow pea stir Fry, flat tasting flat noodles ( even my 12 yr old asked where the spice went) with poor quality pork, and a plate of slightly soggy fried squid.")
    review_list.append("The sauce for the mussels had good flavor, but the mussels themselves didn't taste fresh at all and were of poor quality.")
    review_list.append("Duck confit $20 was dry as a bone awful")


    demo_text = 'Yesterday dumb Bob destroyed my fancy iPhone in beautiful Denver, Colorado. I guess I will have to head over to the Apple Store and buy a new one.'
    demo_url = 'http://www.yelp.com/biz/shanghai-taste-rockville'

    alchemyapi = AlchemyAPI()
    response = alchemyapi.entities('text', demo_text, {'sentiment': 1})

    if response['status'] == 'OK':
        print('## Response Object ##')
        print(json.dumps(response, indent=4))
    
        print('')
        print('## Entities ##')
        for entity in response['entities']:
            print('text: ', entity['text'].encode('utf-8'))
            print('type: ', entity['type'])
            print('relevance: ', entity['relevance'])
            print('sentiment: ', entity['sentiment']['type'])
            if 'score' in entity['sentiment']:
                print('sentiment score: ' + entity['sentiment']['score'])
            print('')
    else:
        print('Error in entity extraction call: ', response['statusInfo'])
        
    for review in review_list:

        print(review)
        response = alchemyapi.entities('text', review, {'sentiment': 1})

        if response['status'] == 'OK':
            print('## Response Object ##')
            print(json.dumps(response, indent=4))
        
            print('')
            print('## Entities ##')
            for entity in response['entities']:
                print('text: ', entity['text'].encode('utf-8'))
                print('type: ', entity['type'])
                print('relevance: ', entity['relevance'])
                print('sentiment: ', entity['sentiment']['type'])
                if 'score' in entity['sentiment']:
                    print('sentiment score: ' + entity['sentiment']['score'])
                print('')
        else:
            print('Error in entity extraction call: ', response['statusInfo'])        
        
        #next steps, how to make it better for yelp results. Perhaps need to train? needs a corpus or lexicon

    response = alchemyapi.keywords('url', demo_url, {'sentiment': 1})
    print('Processing url: ', demo_url)
    
    if response['status'] == 'OK':
        print('## Response Object ##')
        print(json.dumps(response, indent=4))
    
        print('')
        print('## Keywords ##')
        for keyword in response['keywords']:
            if keyword['sentiment']['type'] == 'positive':
                print('text: ', keyword['text'].encode('utf-8'))
                print('relevance: ', keyword['relevance'])
                print('sentiment: ', keyword['sentiment']['type'])
                if 'score' in keyword['sentiment']:
                    print('sentiment score: ' + keyword['sentiment']['score'])
                print('')
    else:
        print('Error in keyword extaction call: ', response['statusInfo'])
        
    demo_url = 'http://www.yelp.com/biz/china-chilcano-washington'
    
    response = alchemyapi.keywords('url', demo_url, {'sentiment': 1})
    print('Processing url: ', demo_url)
    
    if response['status'] == 'OK':
        print('## Response Object ##')
        print(json.dumps(response, indent=4))
    
        print('')
        print('## Keywords ##')
        for keyword in response['keywords']:
            if keyword['sentiment']['type'] == 'positive':
                print('text: ', keyword['text'].encode('utf-8'))
                print('relevance: ', keyword['relevance'])
                print('sentiment: ', keyword['sentiment']['type'])
                if 'score' in keyword['sentiment']:
                    print('sentiment score: ' + keyword['sentiment']['score'])
                print('')
    else:
        print('Error in keyword extaction call: ', response['statusInfo'])
    
    demo_url = 'http://www.yelp.com/biz/ristorante-bonaroti-vienna'
    
    response = alchemyapi.keywords('url', demo_url, {'sentiment': 1})
    print('Processing url: ', demo_url)
    
    if response['status'] == 'OK':
        print('## Response Object ##')
        print(json.dumps(response, indent=4))
    
        print('')
        print('## Keywords ##')
        for keyword in response['keywords']:
            if keyword['sentiment']['type'] == 'positive':
                print('text: ', keyword['text'].encode('utf-8'))
                print('relevance: ', keyword['relevance'])
                print('sentiment: ', keyword['sentiment']['type'])
                if 'score' in keyword['sentiment']:
                    print('sentiment score: ' + keyword['sentiment']['score'])
                print('')
    else:
        print('Error in keyword extaction call: ', response['statusInfo'])
        
    # yelp gets their menu from their website    
    demo_url = 'http://www.yelp.com/menu/ristorante-bonaroti-vienna/dinner-menu'
    
    #First get all the menu items
    demo_url = 'http://washingtondc.menupages.com/restaurants/ristorante-bonaroti/menu'
    response = alchemyapi.entities('url', demo_url, {'sentiment': 1,'showSourceText':1})
    print('Processing url: ', demo_url)
    
    menu_items =[];
    
    if response['status'] == 'OK':
        print('## Response Object ##')
        print(json.dumps(response, indent=4))
    
        print('')
        print('## Keywords ##')
        for entities in response['entities']:
            if entities['sentiment']['type'] == 'positive':
                print('text: ', entities['text'].encode('utf-8'))
                
                menu_items.append(entities['text']);
                
                print('relevance: ', entities['relevance'])
                print('sentiment: ', entities['sentiment']['type'])
                if 'score' in entities['sentiment']:
                    print('sentiment score: ' + entities['sentiment']['score'])
                print('')
    else:
        print('Error in keyword extaction call: ', response['statusInfo'])
    
    demo_url = 'http://www.yelp.com/menu/ristorante-bonaroti-vienna/dinner-menu'    
    
    # then do targeted sentiment on the menu items
    for item in menu_items:
        print(item)
        
        response = alchemyapi.sentiment_targeted('url', demo_url,item, {'sentiment': 1})
        print('Processing url: ', demo_url)
        
        if response['status'] == 'OK':
            print('## Response Object ##')
            print(json.dumps(response, indent=4))
        
            print('')
            print('## Keywords ##')
            for keyword in response['sentiment_targeted']:
                if keyword['sentiment']['type'] == 'positive':
                    print('text: ', keyword['text'].encode('utf-8'))
                    print('relevance: ', keyword['relevance'])
                    print('sentiment: ', keyword['sentiment']['type'])
                    if 'score' in keyword['sentiment']:
                        print('sentiment score: ' + keyword['sentiment']['score'])
                    print('')
        else:
            print('Error in keyword extaction call: ', response['statusInfo'])
        
    
        
    
    #This is pretty solid, can i host this on github?