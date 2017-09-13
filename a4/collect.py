import csv
import sys
import time
from TwitterAPI import TwitterAPI
import conf
import oauth2 as oauth
import json
import pickle
import collections
import re

consumer_key = 'VdymTQ8WVleZQM7z5si879AcG'
consumer_secret = '6N7bkzxue4sT6L3PyCTYCue57AKWuUcaJ57iyJE6Yqf4Mjwf6q'
access_token = '738905027574693888-gnj2kEFEUbc1LVX8amtnEnLVZIohjhA'
access_token_secret = 'sEbG2RaJFRSP9EKunnGkD39swF1gLbIHyLoa0VNA2vLNc'


# This method is done for you. Make sure to put your credentials in thepip install python_twitter file twitter.cfg.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you enteraed above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

    
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

            
            
      
def get_followers(twitter,screen_name,filename,count):
    """ gets  followers of the given screen name and store in the file 
    Args:
      twitter .... A TwitterAPI object.
      screen_name ... 
      filename ..... 
      count .. 
    Returns:
      list of follower ids
    """
    url = 'https://api.twitter.com/1.1/followers/ids.json?&screen_name=@'+screen_name+'&skip_status=true&include_user_entities=false&count='+str(count) 
    consumer = oauth.Consumer(key=consumer_key, secret=consumer_secret)
    access = oauth.Token(key=access_token, secret=access_token_secret)
    client = oauth.Client(consumer, access)
    try:
        response,data = client.request(url)
        dataStr = data.decode('utf-8')    
        if('Rate limit exceeded' in dataStr ):
            print('rate limit exceeded error.. sleep for 15 min')
            time.sleep(61 * 15)
            response,data = client.request(url)
         
        jsonid = json.loads(dataStr)
        li = list(jsonid['ids'])
        output = open(filename, 'wb')
        pickle.dump(li, output)
        output.close()
    except:
        pass
       
    return li   
    
def get_friends(twitter,userid,count):
    """ gets  friends of the given userid
    Args:
      twitter .... A TwitterAPI object.
      userid ... 
      filename ..... 
      count .. 
    Returns:
      list of friend ids
    """
    url = 'https://api.twitter.com/1.1/friends/ids.json?&user_id='+str(userid)+'&skip_status=true&include_user_entities=false&count='+str(count) 
    consumer = oauth.Consumer(key=consumer_key, secret=consumer_secret)
    access = oauth.Token(key=access_token, secret=access_token_secret)
    client = oauth.Client(consumer, access)
    li=[]
    try:
        response,data = client.request(url)
        dataStr = data.decode('utf-8')    
        if('Rate limit exceeded' in dataStr ):
            print('rate limit exceeded error.. sleep for 15 min')
            time.sleep(61 * 15)
            response,data = client.request(url)
         
        jsonid = json.loads(dataStr)
        li = list(jsonid['ids'])
    
    except:
        pass
       
    return li  
    

def get_follower_friends(twitter,followersFile,destfile,friends_count):
    """ gets  friends of the given userid
    Args:
      twitter .... A TwitterAPI object.
      followersFile ... 
      destfile ..... 
    Returns:
      list of friend ids
    """
    allFriends=[]
    followersFriendList=[]
    followersList=[]

    try:
        pkl_file = open(followersFile, 'rb')
        followersList = pickle.load(pkl_file)
        pkl_file.close()
    except:
        pass
    try:
        pkl_file = open(destfile, 'rb')
        followersFriendList = pickle.load(pkl_file)
        pkl_file.close()
    except:
        pass
    for cnt,followerid in enumerate(followersList):
        idList=get_friends(twitter,followerid,friends_count)
        folDict={}
        folDict[followerid]=idList
        followersFriendList.append(folDict)
        output = open(destfile, 'wb')
        pickle.dump(followersFriendList, output)
        output.close() 
    #checking total friends of followers    
    
    try:  
        pkl_file = open(destfile, 'rb')
        trump_fol_friend100 = pickle.load(pkl_file)
        pkl_file.close()   
        for li in  trump_fol_friend100:
            for key,val in li.items():
                for v in val:
                    allFriends.append(v) 
    except:
        pass
    
    return followersFriendList,allFriends         
        
def get_last100_tweets_from_time_line(twitter,screen_name,filename,since_id):
    params = {'screen_name': screen_name,'count':100,'since_id':since_id}
    resource = 'statuses/user_timeline'
    trumpTweetsResponse = robust_request(twitter, resource, params, max_tries=2)
    allTweets=[]

    for r in trumpTweetsResponse:        
        arr=[]
        arr.append(r['id_str'])
        arr.append(r['text'])
        arr.append(0)        
        allTweets.append(arr)
            
    with open(filename, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f,delimiter=',')
        writer.writerow(('ID','DATA','SENTIMENT'))  # only if you want it
        for t in allTweets:
            writer.writerow(t) 
  
    return allTweets
                
            
def get_userid(twitter,screen_name):
    resource = 'users/lookup'
    params = {'screen_name': screen_name}
    user_list_response = robust_request(twitter, resource, params, max_tries=2)
    for u in user_list_response:  
        #print(u['screen_name']) 
        #print(u['id_str'])
        return u['id_str']    
 

def get_screen_name(twitter,userid):
    resource = 'users/lookup'
    params = {'user_id': userid}
    user_list_response = robust_request(twitter, resource, params, max_tries=2)
    for u in user_list_response:  
        #print(u['screen_name']) 
        return u['screen_name']   


def get_five_common_friends_other_than_trump(trumpid):
    
    trump_fol_friend=[]
    try:
     
        pkl_file = open(conf.followers_friends_file, 'rb')
        trump_fol_friend = pickle.load(pkl_file)
        pkl_file.close()
    except:
        pass
  
    
    allfr=[]                
    for li in  trump_fol_friend:
        for key,val in li.items():
            for v in val:
                if(v!=int(trumpid)):
                    allfr.append(v)
                    
                   
    cnt = collections.Counter(allfr)      
    return list(cnt.most_common(5))
    
    
    
    
def main():
    
    twitter = get_twitter()
    trumpid=get_userid(twitter,conf.screen_name)
    output = open(conf.userid_file, 'wb')
    pickle.dump(trumpid, output)
    output.close() 
    allTweets=get_last100_tweets_from_time_line(twitter,conf.screen_name,conf.last_100_tweets_file,conf.since_tweet_id)
    followers_list=get_followers(twitter,conf.screen_name,conf.followers_file,conf.number_of_followers)
    trump_fol_fr,allfriends=get_follower_friends(twitter,conf.followers_file,conf.followers_friends_file,conf.number_of_followers_friends) 
    commonfr=get_five_common_friends_other_than_trump(trumpid)  
    str1=('Number of new tweets collected for test %d'%(len(allTweets)))
    str2=('Number of followers of Trump collected %d'%(len(followers_list)))
    str3=('Number of friends of Trump\'s followers collected %d'%(len(allfriends)))
    str4=('Total number of users collected %d'%(len(allfriends)+len(followers_list)))
    str5=('Top five common friends of Trump\'s followers')
    for f  in commonfr:
        sc=get_screen_name(twitter,f)
        #print(sc)
     
    str7 = 'Number of new tweets collected for test %d'%(len(allTweets))+'\n'
    str8 = 'Number of followers of Trump collected %d'%(len(followers_list))+'\n'   
    f = open(conf.collect_log,'w')
    f.write(str1+'\n'+str2+'\n'+str3+'\n'+str4+'\n'+str5+'\n')
    for fol  in commonfr:
        sc=get_screen_name(twitter,fol)
        f.write(sc)
    f.write('\n')
    f.write(str7+'\n'+str8+'\n')
    f.close()
    
    
if __name__ == '__main__':
    main()