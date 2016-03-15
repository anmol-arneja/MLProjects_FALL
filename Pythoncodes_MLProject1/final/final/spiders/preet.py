# -*- coding: utf-8 -*-
from scrapy import Spider
from scrapy.spiders import CrawlSpider,Rule
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
import urllib
from final.items import FinalItem
tot_links=[]
class PreetSpider(CrawlSpider):
    name = "preet"
    allowed_domains = ["www.reddit.com"]
    start_urls = ['http://www.reddit.com/r/worldnews']
    rules = [Rule(LinkExtractor(allow=['/r/worldnews/\?count=\d*&after=\w*']),callback='parse_item',
    		follow=True)]



    def parse_item(self, response):
        print "Scrapping www.reddit.com for Applied Machine Learning project 1"
        a= response.url
        print a
        htmlfile = urllib.urlopen(a)
        htmltext = htmlfile.read()
        soup = BeautifulSoup(htmltext,"lxml")
        all_links = soup.find_all('a',attrs={'class':'title may-blank '},)
        all_comments = soup.find_all('a',attrs={'class':'comments may-blank'})
        get_date = soup.find_all("time")
        votes = soup.find_all("div",attrs ={'class':'score unvoted'})

        #print len(all_links)
        for i in range(len(all_links)):
            item = FinalItem()
            try:
                url = all_links[i].get("href")
                item['url'] = url
                tot_links.append(url)
                num_links = len(tot_links)
                comments = all_comments[i].text.strip("comments")
                item['num_comments'] = comments
                days = get_date[i].get("title")
                day = days.split()
                day = day[0]
                #print day
                if (day=='Mon' or day =='Tue' or day =='Wed' or day=='Thu' or day=='Fri'):
                    item['weekday'] = 1
                else:
                    item['weekday'] = 0
                if (day=='Sat' or day=='Sun'):
                    item['weekend'] = 1
                else:
                    item['weekend'] = 0
                vote = votes[i].text
                item['votes'] = vote


            except:
                print "URL cannot be opened"
            try:
                htmlfile_child = urllib.urlopen(url)
                htmltext_child = htmlfile_child.read()
                soup_child = BeautifulSoup(htmltext_child,"lxml")
                title = soup_child.title.string

                if len(title)>0:
                    #item ['title'] = title
                    words_title = title.split()
                    num_words_title = len(words_title)
                    #print num_words_title
                    item['num_words_title'] = num_words_title
                    all_meta = soup_child.find_all("meta")
                    for meta in all_meta:
                        #name = meta.get("name")
                        prop = meta.get("property")
                        #print prop
                        #itemm = meta.get("itemprop")
                        if prop == "og:description":
                            keywords = meta.get("content")
                            num_keywords = len(keywords.split())
                            item['num_keywords'] = num_keywords
                    images = soup_child.find_all("img")
                    item['num_images'] = len(images)
                    ahrefs = soup_child.find_all("a")
                    for hrefs in ahrefs:
                        num_hrefs = hrefs.get("href")
                    item['num_hrefs'] = len(num_hrefs)
            except:
                print "I cannot open"
            yield item






