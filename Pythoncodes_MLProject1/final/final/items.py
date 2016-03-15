# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

from scrapy import Item,Field


class FinalItem(Item):
    #image_urls = Field()
    #images = Field()
    url = Field()
    #title = Field()
    num_words_title = Field()
    num_keywords = Field()
    num_comments = Field()
    num_images = Field()
    num_hrefs = Field()
    weekday = Field()
    weekend = Field()
    votes = Field()
    #comment = Field()
    #url = Field()
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass
