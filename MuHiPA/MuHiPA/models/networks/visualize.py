from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
import os
import torch.nn.functional as F
import math



class Visualize():
    def __init__(self, on=False, path= 'data/vqa/coco/raw/', wid_to_word=None):
        self.on = on
        self.path = path
        self.wid_to_word = wid_to_word
        self.batch = None
        self.argmax_original = None
        self.q_type_dic = {"what is" : 0, "is this":1, "how many": 2,
                      "what time" : 3, "are there any" : 4,
                     "what is this" : 5, "what is the": 6,
                     "what color is the":7, "is there a": 8,
                     "none of the above": 9, "what kind of": 10,
                     "what" : 11, "what is in the": 12,
                     "are they": 13, "what type of":14,
                     "is this a":15, "how many people are": 16,
                     "where are the": 17, "what does the": 18,
                     "do you": 19, "is this person": 20, "was":21,
                     "why is the": 22, "what color is": 23,
                     "how": 24, "what room is": 25, "do" :26,
                     "is the": 27, "are they":28, "how many people are in":29,
                     "are there": 30, "are these": 31, "is he": 32,
                     "what color are the": 33, "what is the name":34,
                     "can you" : 35, "which":36, "what animal is": 37,
                     "what is on the" : 38, "is the person": 39,
                     "what sport is" : 40, "what is in the":41,
                     "why" : 42, "is this a" : 43, "is the woman" : 44,
                     "is that a": 45, "are":46, "what color" : 47,
                     "is it" : 48, "what is the person": 49,
                     "is there" : 50,  "where is the" : 8,
                     "what" : 51, "is there a": 52, "does this": 53,
                     "what is the man":54, "what are the" : 55,
                     "what is the" : 56, "what number is" : 57,
                     "is":58, "what are":59, "is the man":60, "what":61,
                     "is this an": 62, "are the" : 63, "what is the woman":64,
                     "does the" : 65, "what brand" : 66, "what is this":67,
                     "who is" : 68, "what is the color of the": 69, "has": 70,
                     "could":71
     
                }  
    
    def multi_image(self, img):
        plt.figure()
        
        images = []
        paths = []
        for i in range(len(img)):

            if 'val2014' in img[i]:
                paths.append(self.path + 'val2014/')
            if 'train2014' in img[i]:
                paths.append(self.path + 'train2014/')
            
            images.append(plt.imread(os.path.join(paths[i], img[i])))

        #subplot(r,c) provide the no. of rows and columns
        x= 0
        y =0
        for i in range(2, len(img)+1):
            if len(img) % i == 0:
                x = i
                break
        print(x," ", y, " ", len(img))  
        y = int(len(img) / x)
        print(x," ", y, " ", len(img))  
        f, axarr = plt.subplots(x,y) 

        # use the created array to output your multiple images. In this case I have stacked 4 images vertically
        k = 0
        for i in range(x):
            for j in range(y):
                # print(i,y)
                axarr[i][j].imshow(images[k])
                axarr[i][j].set_axis_off()
                k += 1
        # axarr[0][0].imshow(images[0])
        # axarr[0][1].imshow(images[1])
        # axarr[0][2].imshow(images[2])
        # axarr[1][0].imshow(images[3])
        # axarr[1][1].imshow(images[4])
        # axarr[1][2].imshow(images[5])
        # axarr[2][0].imshow(images[6])
        # axarr[2][1].imshow(images[7])
        # axarr[2][2].imshow(images[8])
        plt.axis('off')
        plt.title("")
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show()
        
    def cosine_similarity_reas_agg(self, reas, agg, title ="", np_ = False):
        if not np_:
            agg = agg.cpu().detach().numpy().tolist()
            reas = reas.cpu().detach().numpy().tolist()
        similarity_agg = []
        similarity_reas = []
        labels = []
        
        from scipy import spatial

        for i in range(len(agg)):
            agg_1 = agg[i]
            reas_1 = reas[i]
            similarity_agg_item = []
            similarity_reas_item = []
            for j in range(len(agg)):
                ex_agg = agg[j]
                ex_reas = reas[j]
                similarity_agg_item.append(float("{0:.2f}".format(1 - spatial.distance.cosine(agg_1, ex_reas))))
                similarity_reas_item.append(float("{0:.2f}".format(1 - spatial.distance.cosine(reas_1, ex_agg))))
            similarity_agg.append(similarity_agg_item)
            similarity_reas.append(similarity_reas_item)
            labels.append(str(i))
            

        # text = ""
        # for i in range(len(q_txt)):
        #     text += "\n" + str(i) + ")" + str(q_txt[i])
        # seperator = self.seperator("-------q_txt")
        # self.showText(text, seperator)
        
        
        similarity_agg = np.array(similarity_agg)

        fig, ax = plt.subplots()
        cbarlabel = "similarity_agg_ " + title
        im, cbar = self.heatmap(similarity_agg, labels, labels, ax=ax,
                           cmap="YlGn", cbarlabel=cbarlabel)
        texts = self.annotate_heatmap(im, valfmt="{x:.1f}")
        
        fig.tight_layout()
        plt.show()
        
        
        # similarity_reas = np.array(similarity_reas)

        # fig, ax = plt.subplots()
        # cbarlabel = "similarity_reas_ " + title
        # im, cbar = self.heatmap(similarity_reas, labels, labels, ax=ax,
        #                    cmap="YlGn", cbarlabel=cbarlabel)
        # texts = self.annotate_heatmap(im, valfmt="{x:.1f}")
        
        # fig.tight_layout()
        # plt.show()

    def cosine_similarity(self, m_reas, m_agg, title="", np_ = False):
        if not np_:
            m_agg = m_agg.cpu().detach().numpy().tolist()
            m_reas = m_reas.cpu().detach().numpy().tolist()
        similarity_agg = []
        similarity_reas = []
        labels = []
        
        from scipy import spatial

        for i in range(len(m_agg)):
            m_agg_1 = m_agg[i]
            m_reas_1 = m_reas[i]
            similarity_agg_item = []
            similarity_reas_item = []
            for j in range(len(m_agg)):
                ex_agg = m_agg[j]
                ex_reas = m_reas[j]
                similarity_agg_item.append(float("{0:.2f}".format(1 - spatial.distance.cosine(m_agg_1, ex_agg))))
                similarity_reas_item.append(float("{0:.2f}".format(1 - spatial.distance.cosine(m_reas_1, ex_reas))))
            similarity_agg.append(similarity_agg_item)
            similarity_reas.append(similarity_reas_item)
            labels.append(str(i))
            

        # text = ""
        # for i in range(len(q_txt)):
        #     text += "\n" + str(i) + ")" + str(q_txt[i])
        # seperator = self.seperator("-------q_txt")
        # self.showText(text, seperator)
        
        
        similarity_agg = np.array(similarity_agg)

        fig, ax = plt.subplots()
        cbarlabel = "similarity_agg_" + title
        im, cbar = self.heatmap(similarity_agg, labels, labels, ax=ax,
                           cmap="YlGn", cbarlabel=cbarlabel)
        texts = self.annotate_heatmap(im, valfmt="{x:.1f}")
        
        fig.tight_layout()
        plt.show()
        
        
        similarity_reas = np.array(similarity_reas)

        fig, ax = plt.subplots()
        cbarlabel = "similarity_reas_" + title
        im, cbar = self.heatmap(similarity_reas, labels, labels, ax=ax,
                           cmap="YlGn", cbarlabel=cbarlabel)
        texts = self.annotate_heatmap(im, valfmt="{x:.1f}")
        
        fig.tight_layout()
        plt.show()

    def distance(self, q_reas, q_agg, title= "", np_=False):
        if not np_:
            q_agg = q_agg.cpu().detach().numpy().tolist()
            q_reas = q_reas.cpu().detach().numpy().tolist()
        similarity_agg = []
        similarity_reas = []
        labels = []
        
        from scipy import spatial

        for i in range(len(q_agg)):
            q_agg_1 = q_agg[i]
            q_reas_1 = q_reas[i]
            similarity_agg_item = []
            similarity_reas_item = []
            for j in range(len(q_agg)):
                ex_agg = q_agg[j]
                ex_reas = q_reas[j]
                import numpy
                dist1 = numpy.linalg.norm(q_agg_1-ex_agg)
                dist2 = numpy.linalg.norm(q_reas_1-ex_reas)
                similarity_agg_item.append(float("{0:.2f}".format(1 - dist1)))
                similarity_reas_item.append(float("{0:.2f}".format(1 - dist2)))
            similarity_agg.append(similarity_agg_item)
            similarity_reas.append(similarity_reas_item)
            labels.append(str(i))
            

        # text = ""
        # for i in range(len(q_txt)):
        #     text += "\n" + str(i) + ")" + str(q_txt[i])
        #     # print("::::::::::::::::::::::")
        # seperator = self.seperator("-------q_txt")
        # self.showText(text, seperator)
        
        cbarlabel = "distance_agg" + title
        similarity_agg = np.array(similarity_agg)

        fig, ax = plt.subplots()
        
        im, cbar = self.heatmap(similarity_agg, labels, labels, ax=ax,
                           cmap="YlGn", cbarlabel=cbarlabel)
        texts = self.annotate_heatmap(im, valfmt="{x:.1f}")
        
        fig.tight_layout()
        plt.show()
        
        
        similarity_reas = np.array(similarity_reas)

        fig, ax = plt.subplots()
        cbarlabel = "distance_reas" + title
        im, cbar = self.heatmap(similarity_reas, labels, labels, ax=ax,
                           cmap="YlGn", cbarlabel=cbarlabel)
        texts = self.annotate_heatmap(im, valfmt="{x:.1f}")
        
        fig.tight_layout()
        plt.show()
                
        
        
        
        
    def sigmoid(self, xlist):
        return [1 / (1 + math.exp(-x)) for x in xlist]
        
    def setBatch(self, batch):
        self.batch = batch
        
    def seperator(self, s=""):
        if(self.on):
            print("---------------------------", s)
            
    def set_argmax_image(self, argmax):
        self.argmax_image = argmax[0]
        
    def set_argmax_question(self, argmax):
        self.argmax_question = argmax[0]          
    
    def set_argmax_tag(self, argmax):
        self.argmax_tag = argmax[0]       
     
    def question_type(self, batch=None, q_type=None, seperator=""):
        if(self.on):
            q = batch['question']
            qq = q[0].cpu().detach().numpy()
            _ ,q_type = q_type.max(1)
            q_type = q_type[0].cpu().detach().numpy()

            q_type_dic = {v: k for k, v in self.q_type_dic.items()}
            q_type = q_type_dic[int(q_type)]
            q_type_target = batch['question_type'][0]
    
            text = ""
            text += "  Question Type target \n" + q_type_target + "\n  Question Type predict \n" +  q_type + "\n"
            print("---------- Question -----------------")
            
            self.wid_to_word[0] = " "
            words = [self.wid_to_word[qq[i]] for i in range (qq.shape[0])]

            text += "\n" + str(words)
            
    
            self.showText(text, seperator)
            
    def tag(self, batch=None, features=None, argmax=None, seperator=""):
        if(self.on):
            text = ""
            text += seperator
            print("-------------", seperator)
            if(batch == None):
                batch=self.batch
            tag = batch['cls_text']
            tag = tag[0]

    
        
            text += "Tag "
            print("---------- Tag -----------------")
            

            words = [str(tag[i]) for i in range (len(tag))]
            words1 =  [words[i] for i in range (len(words)//2)]
            words2 =  [words[i+len(words)//2] for i in range (len(words)//2)]
            text1 = ""
            text2 = ""
            # text1 += text + "\n" + str(words1)
            # text2 += text + "\n" + str(words2)
            text1 += text + "\n" 
            text2 += text + "\n" 
            print(words)
            # print("fffffffffffffffffffffffffffffffffffff")
            # print(features.shape)
            if (argmax == None):
                _, argmax = features.max(1)
                f, _ = features.max(2)
                argmax = argmax[0]
                f = f[0]

                # print(f.shape)
                # print(argmax.shape)
            else:
                f, _ = features.max(2)
                f = f[0]
        
            armx = [0] * 36
            # print("-----------------------------88888888888888877777777777777")
            # print(argmax)
            
            
            for i in range(argmax.shape[0]):
                # print(argmax)
                # print(argmax[i])
                idx = argmax[i].cpu().detach().numpy()
                # if (idx == 35):
                    # if (features[0,35,i] == 0):
                        
                    # print(features[0,:,i])
                armx[idx] = armx[idx] + 1
            # print(armx)
            count = 0
            for i in range(len(words1)):
                count += 1
                # if (count>12):
                #     break
                s1 = f[i].cpu().detach().numpy()
                s2 = armx[i]/argmax.shape[0]
                # print(armx[i])
                # print(argmax.shape[0])
                # print(s2)
                s = " max:" + ("%.2f" % s1) + " argmax:" + ("%.2f" % s2)
                text1 += "\n" + str(i) + ":" + str(words1[i]) + s
                print(i, ":", words1[i], s)
            
            self.showText(text1, seperator)
            for i in range(len(words2)):
                j = i
                i = i+len(words)//2
                count += 1
                # if (count>12):
                #     break
                s1 = f[i].cpu().detach().numpy()
                s2 = armx[i]/argmax.shape[0]
                # print(armx[i])
                # print(argmax.shape[0])
                # print(s2)
                s = " max:" + ("%.2f" % s1) + " argmax:" + ("%.2f" % s2)
                text2 += "\n" + str(i) + ":" + str(words2[j]) + s
                print(i, ":", words2[j], s)
            
            self.showText(text2, seperator)
        
    def argmax_change_question(self, batch=None, features=None, title="", bert=False):
        if(self.on): 
            if(batch == None):
                batch=self.batch
                
            f = features[0]
            
            
            armx_sum = [0] * 36      
            total = f.sum(0)
            minimum = f.min(0)[0]
            total_sum = 0
            for i in range(self.argmax_question.shape[0]):
                idx = self.argmax_question[i].cpu().detach().numpy()
                armx_sum[idx] = armx_sum[idx] + f[i] + abs(minimum)
                total_sum += abs(f[i]) + abs(minimum)
                
               
            for i in range(len(armx_sum)):
                if (not isinstance(armx_sum[i], int)):
                    armx_sum[i] = armx_sum[i].cpu().detach().numpy()
            # print(armx)
            armx = armx_sum
            sig = np.array(armx)
            # sig = sig / total_sum #* 1000
            # sig = (sig-min(sig))/(max(sig)-min(sig))
          
            
     
            
            
            text = ""
            q = batch['question']
            qq = q[0].cpu().detach().numpy()
            text += "  Question"         
            self.wid_to_word[0] = " "
            words = [self.wid_to_word[qq[i]] for i in range (qq.shape[0])]
            if(bert):
                words.insert(0, "[CLS]")
                words.append("[SEP]")
            text += "\n" + str(words)

            count = 0
            
         
            
            
            
            for i in range(len(words)):
                count += 1
                if (count>12):
                    break
                # s3 = armx[i]/total_sum
                s3 = sig[i] /total_sum
                

                s = " sum of argmax:" + ("%.2f" % s3)
                text += "\n" + str(i) + ":" + str(words[i]) + s
                print(i, ":", words[i], s)
            
            self.showText(text, title)
    
    def argmax_change_image(self, batch=None, features=None, title=""):
        if(self.on):
            # self.seperator(title)

            
            if(batch == None):
                batch=self.batch
                
            
            
            f = features[0]
            # f2 = features2[0]
            # f = f1 - f2
            
            
            armx_sum = [0] * 36
            print("-----------------------------88888888888888877777777777777")
            # print(argmax)
            
            total = f.sum(0)
            minimum = f.min(0)[0]
            
            print(minimum)
            total_sum = 0
            for i in range(self.argmax_image.shape[0]):
                # print(argmax)
                # print(argmax[i])
                idx = self.argmax_image[i].cpu().detach().numpy()
                # if (idx == 35):
                    # print(features[0,:,i])
                armx_sum[idx] = armx_sum[idx] + f[i] + abs(minimum)
                
                total_sum += abs(f[i]) + abs(minimum)
            
            for i in range(len(armx_sum)):
                if (not isinstance(armx_sum[i], int)):
                    armx_sum[i] = armx_sum[i].cpu().detach().numpy()
            # print(armx)
            armx = armx_sum
            
            
            
            # armx
            sig = np.array(armx)
            sig = sig  #* 1000
            # sig = (sig-min(sig))/(max(sig)-min(sig))
            # sig = np.mean(sig).mean
            
            # sig = sig.tolist()
            # sig = self.sigmoid(sig)
            
            
            image_name = batch['image_name'][0]

            visual = batch['visual'][0]
            coord = batch['norm_coord'][0]

            coord = coord.cpu()
            visual = visual.cpu()

            if 'val2014' in image_name:
                path = self.path + 'val2014/'
            if 'train2014' in image_name:
                path = self.path + 'train2014/'
            image = plt.imread(os.path.join(path, image_name))
            h, w, d = image.shape
            # print(w)
            # print(h)
            # print(d)
            
            figsize=(10,6)
            if figsize != None:
                plt.figure(figsize=figsize)
                """Imshow for Tensor."""
                plt.imshow(image)
        
            count = 0
            plt.gca().add_patch(Rectangle((0,0),w,h, linewidth=1,facecolor='k', alpha=.4))
            
            
            maxi = 0
            for i in range(sig.size):
                if (sig[i]> maxi):
                    maxi = sig[i]
            
            
            ran = np.random.randint(0, high=36, size=1, dtype=int)
            
            
            for i in range(coord.shape[0]):
                if(True):
                    # print("--------------------------------------------------------------")
                    edgecolor = (((i+4)%9)/10, ((i+8)%7)/10, ((i+16)%8)/10)
                    # print("--------------------------edggggggggggggggeee")

                    font = {'family': 'serif',
                            'color':  edgecolor,
                            'weight': 'normal',
                            'size': 16,
                            }
                    count = count + 1
                    # print(armx[i])
                    # print("kiuuuuuuuuuuuuuuuuuuuuuujjjjjjjjjjjjjjjjj")
                    
                    facecolor = 'r'
                    
                    # a =  svisual
                    # a =  a*2 if (a*2 < 1) else 0.8
                    
                    if ( ran == i):
                        # a = sig[i] / total_sum
    
                        # a =  a*2 if (a*2 < 1) else 0.6
                        a = 0.8
    
                            
                        
                        # s3 = armx[i]/total_sum
                        # if (s3*3 < 1 and s3*3>0 ):
                        #     a = s3*3
                        #     facecolor = 'r'
                        # elif(s3*3>1):
                        #     a = 1
                        #     facecolor = 'r'
                        # # elif(abs(s3)*2>1):
                        # #     a = 1
                        # #     facecolor = 'g'
                        # else:
                        #     # a = abs(s3)*2
                        #     a = 0
                        #     # facecolor = 'g'
                        #     #a = 0         
                        # # print("aaaaaaaaaaaaaaaaaaaaaaaa")
                        # # print(total)
                        # # print(a)
                        
                        
                        #  svisual = visual[i]
                        # stags = tags[i]
    
                        # a =  svisual
                        # a =  a*2 if (a*2 < 1) else 0.8
                        
                        # text += "\n" + "cls_text ----"+ str(i) +"="+ str(cls_text[i])+ "==== "+ str(stags)
                        # print("cls_text ----", i,"=", cls_text[i], "==== ", stags)
    
                        plt.gca().add_patch(Rectangle((coord[i][0]*w,coord[i][1]*h),coord[i][2]*w,coord[i][3]*h, linewidth=1,facecolor='w', alpha=a))

                    
            plt.title(image_name+ " " +title)
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.show()
            
    def get_question(self, q):
        self.wid_to_word[0] = ""
        questions = []
        for i in range(q.shape[0]):

            w = q[i].cpu().detach().numpy()
            words = [self.wid_to_word[w[i]] for i in range (w.shape[0])] 
            # words.insert(0, "[CLS]")
            sentence = "[CLS]"
            for j in range(len(words)):
                if (words[j] == ""):
                    words[j] = "[SEP]"
                    sentence += " " + words[j]
                    break
                sentence += " " + words[j]
            # questions.append(str(words))
            questions.append(sentence)
            # print(questions)
        return questions
    
    
    def get_question(self, q):
        qq = q.cpu().detach().numpy()
        text = ""
        self.wid_to_word[0] = " "
        words = [self.wid_to_word[qq[i]] for i in range (qq.shape[0])]
        for i in range(len(words)):
            text += " " + str(words[i])
        return text
    
    def question(self, batch=None, features=None, argmax=None, seperator="", bert=False):
        if(self.on):
            text = ""
            text += seperator
            print("-------------", seperator)
            if(batch == None):
                batch=self.batch
            q = batch['question']
            qq = q[0].cpu().detach().numpy()
    
        
            text += "  Question"
            print("---------- Question -----------------")
            
            self.wid_to_word[0] = " "
            words = [self.wid_to_word[qq[i]] for i in range (qq.shape[0])]
            if(bert):
                words.insert(0, "[CLS]")
                words.append("[SEP]")
            text += "\n" + str(words)
            print(words)
            # print("fffffffffffffffffffffffffffffffffffff")
            # print(features.shape)
            if (argmax == None):
                _, argmax = features.max(1)
                f, _ = features.max(2)
                argmax = argmax[0]
                f = f[0]
                # print(f.shape)
                # print(argmax.shape)
            else:
                f, _ = features.max(2)
                f = f[0]
            
            armx = [0] * 36
            # print("-----------------------------88888888888888877777777777777")
            # print(argmax)
            
            
            for i in range(argmax.shape[0]):
                # print(argmax)
                # print(argmax[i])
                idx = argmax[i].cpu().detach().numpy()
                # if (idx == 35):
                    # if (features[0,35,i] == 0):
                        
                    # print(features[0,:,i])
                armx[idx] = armx[idx] + 1
            # print(armx)
            count = 0
            for i in range(len(words)):
                count += 1
                if (count>12):
                    break
                s1 = f[i].cpu().detach().numpy()
                s2 = armx[i]/argmax.shape[0]
                # print(armx[i])
                # print(argmax.shape[0])
                # print(s2)
                s = " max:" + ("%.2f" % s1) + " argmax:" + ("%.2f" % s2)
                text += "\n" + str(i) + ":" + str(words[i]) + s
                print(i, ":", words[i], s)
            
            self.showText(text, seperator)
            
            
        
        
        
    
    def obj(self, batch):
        if(self.on):
            v = batch['visual']
            q = batch['question']
            l = batch['lengths'].data
            c = batch['norm_coord']
            
            self.density_plot(v[0], True)
            self.box_plot(v[0])
    
    def answers(self, batch, out):
        if(self.on):
            q = batch['question']
            qq = q[0].cpu().detach().numpy()
            text = ""
            text += "\n" + "Question:"

            print("---------- Question -----------------")
            
            self.wid_to_word[0] = " "
            words = [self.wid_to_word[qq[i]] for i in range (qq.shape[0])]
            text += "\n" + str(words)
            print(words)
            print("-------------------- Answers ------------------------")
            text += "\n" + "Answers :"
            print("predicted answers:")
            text += "\n" + str(out['answers'][0])
            print(out['answers'][0])
            text += "\n" + "target answers:"
            print("target answers:")
            text += "\n" + str(batch['answer'][0])
            print(batch['answer'][0])
            
            text += "\n" + "Answers Matrix:"
            print("----------- Answers Matrix ---------------")
           
            maximum, _ = out['logits'].max(1)
            minimum, _ = out['logits'].min(1)
            mean = out['logits'].mean(1)
            var = out['logits'].var(1)
            std = out['logits'].std(1)
            
            text += "\n" + "maximum: " +  str(maximum[0]) + "\nminimum: " + str(minimum[0]) + "\nmean: " + str(mean[0]) + "\nvariance: " + str(var[0]) + "\nstd: " + str(std[0])
            print("maximum: ", maximum[0], "\nminimum: ", minimum[0], "\nmean: ", mean[0]
                  , "\nvariance: ", var[0], "\nstd: ", std[0])
            
            self.showText(text, title="Answers")
            self.density_plot(out['logits'][0], title="Answers")
            
    #        batch['answer_id']
            
    #        out['answer_ids']
        
    def add_n_obs(self,df):
        if(self.on):
    #        medians_dict = {grp.median() for grp in df.groupby(group_col)}
            medians = np.median(df,1)
    #        xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
    #        n_obs = df.groupby(group_col)[y].size().values
            for i in range(df.shape[0]):
    #        for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
                plt.text(df[i], medians[i]*1.01, "#obs : "+str(i), horizontalalignment='center', fontdict={'size':14}, color='white')
 
        
    def box_plot(self, df):
        if(self.on):
            df = df.cpu().detach().numpy()
    
     
            # Draw Plot
    #        plt.figure(figsize=(13,10), dpi= 80)
            data = [df[0]]
            for i in range (35):
                data.append(df[i+1])
                
    #            mpl.pyplot.boxplot(df[i], notch=None, vert=None, patch_artist=None, widths=None)
            fig = plt.figure(figsize =(10, 7)) 
      
    #        Creating axes instancereating axes instancereating axes instance 
            ax = fig.add_axes([0, 0, 1, 1]) 
      
    #      Creating plot 
            bp = ax.boxplot(data) 
        # Decoration
            plt.title('Box Plot of Highway Mileage by Vehicle Class', fontsize=22)
            plt.show()
        
    
    def statics(self, feature, title=""):
        if(self.on):
            self.seperator(title)

            maximum, _ = feature.max(1)
            minimum, _ = feature.min(1)
            mean = feature.mean(1)
            var = feature.var(1)
            std = feature.std(1)
            
            print("maximum: ", maximum[0], "\nminimum: ", minimum[0], "\nmean: ", mean[0]
                  , "\nvariance: ", var[0], "\nstd: ", std[0])

            # self.density_plot(out['logits'][0], title="Answers")            
            
    
    def density_plot(self, df, multi=False, title=""):
        if(self.on):

            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
            if (multi):
            # Draw Plot
                plt.figure(figsize=(8,5), dpi= 80)
                for i in range(df.shape[0]):
                    color = (((i+4)%9)/10, ((i+8)%10)/10, ((i+16)%8)/10)
                    print(color)
                    sns.kdeplot(df[i].cpu().detach().numpy(), shade=True, color=color, label=i, alpha=.7)
                # Decoration
                plt.title('Density Plot', fontsize=22)
                plt.legend()
                plt.show()
            else:
                # Draw Plot
                plt.figure(figsize=(8,5), dpi= 80)
                sns.kdeplot(df.cpu().detach().numpy(), shade=True, color="g", label="Cyl", alpha=.7)
                # Decoration
                plt.title("Density Plot "+title, fontsize=22)
                plt.legend()
                plt.show()
        
    def images (self,  batch=None, visual=None, argmax=None, title=""):
        if(self.on):
            if(batch == None):
                batch=self.batch
           
            index = batch['index']
            image_name = batch['image_name']
            if (visual == None):
                visual = batch['visual']
            coord = batch['coord']
            if (argmax== None):
                argmax = visual.max(1)[1]
            coord = batch['norm_coord']
               
    
            # print("index ------------------", index)
            # print("image_name ------------------", image_name)
            self.showImage(image_name[0], coord[0], visual[0], argmax[0], title)    
        
    def showJustImage(self, batch):
        image_name = batch['image_name']
        img_name = image_name[0]
        if(self.on):
            
            print("-------------------------------------------")
            # print(coord)
            
            # img_name = img_name.cpu()

            if "val" in img_name:
                path = self.path + "val2014/"
            elif "train" in img_name:
                path = self.path + "train2014/"
            elif "test" in img_name:
                path = self.path + "test2015/"
            else:
                print(img_name)
                print(type(img_name))
                path = self.path
            image = plt.imread(os.path.join(path, img_name))
            h, w, d = image.shape

            
            figsize=(10,6)
            if figsize != None:
                plt.figure(figsize=figsize)
                """Imshow for Tensor."""
                #inp = inp.numpy().transpose((1, 2, 0))
                #mean = np.array([0.485, 0.456, 0.406])
                #std = np.array([0.229, 0.224, 0.225])
                #inp = std * inp + mean
                #inp = np.clip(inp, 0, 1)
                plt.imshow(image)
            plt.title(img_name)
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.show()   
        
    
    def showImage(self, img_name, coord, visual, argmax,title):
        if(self.on):
            
            # print("-------------------------------------------")
            # print(coord)
            
            # img_name = img_name.cpu()
            # coord = batch['coord']
            coord = coord.cpu()
            if (visual != None):
                visual = visual.cpu()
                
            

            argmax = argmax.cpu()
            if 'val2014' in image_name:
                path = self.path + 'val2014/'
            if 'train2014' in image_name:
                path = self.path + 'train2014/'
            # path = self.path
            image = plt.imread(os.path.join(path, img_name))
            h, w, d = image.shape
            # print(w)
            # print(h)
            # print(d)
            

            armx = [0] * 36
            for i in range(argmax.shape[0]):
                idx = argmax.cpu().detach().numpy()
                armx[idx[i]] += 1
            
            figsize=(10,6)
            if figsize != None:
                plt.figure(figsize=figsize)
                """Imshow for Tensor."""
                #inp = inp.numpy().transpose((1, 2, 0))
                #mean = np.array([0.485, 0.456, 0.406])
                #std = np.array([0.229, 0.224, 0.225])
                #inp = std * inp + mean
                #inp = np.clip(inp, 0, 1)
                plt.imshow(image)
        
            count = 0
            for i in range(coord.shape[0]):
    #        for i in range(3):
    #            plt.gca().add_patch(Rectangle((coord[i][0]*10//width,coord[i][1]*6//height),coord[i][2]*10//width,coord[i][3]*6//height,linewidth=1,edgecolor='r',facecolor='none'))
                # print("cooooooooords")
                # print(coord[i])
                
                # if (coord[i][0]*w+coord[i][2]*w<=w and coord[i][1]*h + coord[i][3]*h <= h):
                if(True):
                    # print("--------------------------------------------------------------")
                    edgecolor = (((i+4)%9)/10, ((i+8)%7)/10, ((i+16)%8)/10)
                    # print("--------------------------edggggggggggggggeee")
                    
                    font = {'family': 'serif',
                            'color':  edgecolor,
                            'weight': 'normal',
                            'size': 16,
                            }
                    count = count + 1
                    # print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                    # print(visual.shape)
                    # print(visual.shape)
                    if (visual != None):
                        s1 = visual[i].max(0)[0].cpu().detach().numpy()
                        s2 = visual[i].mean(0).cpu().detach().numpy()

                    s3 = armx[i]/argmax.shape[0]

                    # print("--------------------------edggggggggggggggeee")
                    # print(s3)
                    s = "" + ("%.2f" % s1) + "_" + ("%.2f" % s2) + "_" + ("%.2f" % s3)
                    # print(s)
                    # plt.text(coord[i][0]*w, coord[i][1]*h, s,  fontdict=font)
                    a =  s3*2 if (s3*2 < 1) else 1
                    plt.gca().add_patch(Rectangle((coord[i][0]*w,coord[i][1]*h),coord[i][2]*w,coord[i][3]*h,linewidth=1,edgecolor=edgecolor,facecolor='r', alpha=a))
            # print(count) 
            plt.title(img_name+ " " +title)
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.show()



    def showEarlyFusion(self, batch=None, visual=None, title="None"):
        if(self.on):
            
            if(batch == None):
                batch=self.batch
           
            coord = batch['norm_coord']
            cls_text = batch['cls_text']
            index = batch['index']
            image_name = batch['image_name']
            if (visual == None):
                visual = batch['visual']

            
                
            tags = visual[:,:,0:1240]
            visual = visual[:,:,1240:2048]
            
            # index=index[0]
            text = " Tags"
            cls_text=cls_text[0]
            image_name=image_name[0]
            tags=tags[0]
            visual = visual[0]
            
            tags = tags
            visual = visual
            
            tags = tags.mean(1).cpu().detach().numpy()
            visual = visual.mean(1).cpu().detach().numpy()
            coord = coord[0].cpu().detach().numpy()
    
            if 'val2014' in image_name:
                path = self.path + 'val2014/'
            if 'train2014' in image_name:
                path = self.path + 'train2014/'
            # path = self.path
            image = plt.imread(os.path.join(path, image_name))
            h, w, d = image.shape

           
            
            figsize=(10,6)
            if figsize != None:
                plt.figure(figsize=figsize)
                """Imshow for Tensor."""
                #inp = inp.numpy().transpose((1, 2, 0))
                #mean = np.array([0.485, 0.456, 0.406])
                #std = np.array([0.229, 0.224, 0.225])
                #inp = std * inp + mean
                #inp = np.clip(inp, 0, 1)
                plt.imshow(image)
        
            count = 0
            plt.gca().add_patch(Rectangle((0,0),w,h, linewidth=1,facecolor='k', alpha=.4))
            for i in range(coord.shape[0]):
    #        for i in range(3):
    #            plt.gca().add_patch(Rectangle((coord[i][0]*10//width,coord[i][1]*6//height),coord[i][2]*10//width,coord[i][3]*6//height,linewidth=1,edgecolor='r',facecolor='none'))
                # print("cooooooooords")
                # print(coord[i])
                
                # if (coord[i][0]*w+coord[i][2]*w<=w and coord[i][1]*h + coord[i][3]*h <= h):
                if(True):
                    # print("--------------------------------------------------------------")
                    edgecolor = (((i+4)%9)/10, ((i+8)%7)/10, ((i+16)%8)/10)
                    # print("--------------------------edggggggggggggggeee")
                    
                    font = {'family': 'serif',
                            'color':  edgecolor,
                            'weight': 'normal',
                            'size': 16,
                            }
                    count = count + 1
                    
                    svisual = visual[i]
                    stags = tags[i]

                    a =  svisual
                    a =  a*2 if (a*2 < 1) else 0.8
                    
                    text += "\n" + "cls_text ----"+ str(i) +"="+ str(cls_text[i])+ "==== "+ str(stags)
                    print("cls_text ----", i,"=", cls_text[i], "==== ", stags)

                    plt.gca().add_patch(Rectangle((coord[i][0]*w,coord[i][1]*h),coord[i][2]*w,coord[i][3]*h, linewidth=1,facecolor='w', alpha=a))

            self.showText(text, title="Tags Attention")
            plt.title(image_name+ " " +title)
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.show()



            
    def showText(self, text = "", title = ""):
        if(self.on):     
            path = '/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/background.png'
            image = plt.imread(path)
            h, w, d = image.shape
            # print(w)
            # print(h)
            # print(d)
            figsize=(10,6)
            plt.figure(figsize=figsize)
            plt.imshow(image)
            edgecolor = (((20+4)%9)/10, ((31+8)%7)/10, ((32+16)%8)/10)
            font = {'family': 'serif',
                    'color':  edgecolor,
                    'weight': 'normal',
                    'size': 15,
                    }
            
            plt.text(50, 300, text,  fontdict=font)
            plt.title(title)
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.show()
            
    def heatmap(self, data, row_labels, col_labels, ax=None,
                cbar_kw={}, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.
    
        Parameters
        ----------
        data
            A 2D numpy array of shape (N, M).
        row_labels
            A list or array of length N with the labels for the rows.
        col_labels
            A list or array of length M with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """
    
        if not ax:
            ax = plt.gca()
    
        # Plot the heatmap
        im = ax.imshow(data, **kwargs)
    
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    
        # We want to show all ticks...
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        # ... and label them with the respective list entries.
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
    
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")
    
        # Turn spines off and create white grid.
        # ax.spines[:].set_visible(False)
    
        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
    
        return im, cbar
    
    
    def annotate_heatmap(self, im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.
    
        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """
    
        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()
    
        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.
    
        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)
    
        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = mpl.ticker.StrMethodFormatter(valfmt)
    
        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
    
        return texts



def distance_cosine (batch_loader, file_name="averaged_dict"):
    count = 0
    maxi = 3
    items = []
    vis_imgs = []
    vis_ques = []
    qq = []
    q = []
    mm = []
    v = []
    cq = []
    cv = []
    ans = []
    pre = []
    vis_ques_type = []
    vvv0 = []
    vvv1 = []
    
    
    an1 = 'grass'
    an2 = 'wood'
    
    for i, batch in enumerate(batch_loader):
        with torch.no_grad():
            out = m.forward2(batch)  
        for j in range(len(batch['answer'])):
        # if 'time' in item['type']:
            if an1 == batch['answer'][j]: #and out['answers'][j] != batch['answer'][j]:
            # if 'is the' in item['type'] and not ('what' in item['type']) and not ('or' in item['type']) and not ('where' in item['type']):
                vis_imgs.append(batch['image_name'][j])
                
                vis_ques_type.append(batch['question_type'][j])
                # with torch.no_grad():
                #     out = m.forward2(batch)  
                vis_ques.append(out['question'][j])
                qq.append(out['q_reas'][j].cpu().detach().numpy().tolist())
                q.append(out['q_agg'][j].cpu().detach().numpy().tolist())
                mm.append(out['v_reas'][j].cpu().detach().numpy().tolist())
                v.append(out['v_agg'][j].cpu().detach().numpy().tolist())
                cq.append(out['cellq'][j].cpu().detach().numpy().tolist())
                cv.append(out['cellv'][j].cpu().detach().numpy().tolist())
                ans.append(batch['answer'][j])
                pre.append(out['answers'][j])
                vvv0.append(out['vvv'][0][j].cpu().detach().numpy().tolist())
                vvv1.append(out['vvv'][1][j].cpu().detach().numpy().tolist())
                count += 1
            if count > maxi:
                print("------------------------------------",i)
                break
        if count > maxi:
                print("------------------------------------",i)
                break
    count = 0
    for i, batch in enumerate(batch_loader):
        with torch.no_grad():
            out = m.forward2(batch)
        for j in range(len(batch['answer'])):
        # if 'time' in item['type']:
            if an2 == batch['answer'][j]:# and out['answers'][j] != batch['answer'][j]:
            # if 'is the' in item['type'] and not ('what' in item['type']) and not ('or' in item['type']) and not ('where' in item['type']):
                vis_imgs.append(batch['image_name'][j])
                
                vis_ques_type.append(batch['question_type'][j])

                vis_ques.append(out['question'][j])
                qq.append(out['q_reas'][j].cpu().detach().numpy().tolist())
                q.append(out['q_agg'][j].cpu().detach().numpy().tolist())
                mm.append(out['v_reas'][j].cpu().detach().numpy().tolist())
                v.append(out['v_agg'][j].cpu().detach().numpy().tolist())
                cq.append(out['cellq'][j].cpu().detach().numpy().tolist())
                cv.append(out['cellv'][j].cpu().detach().numpy().tolist())
                ans.append(batch['answer'][j])
                pre.append(out['answers'][j])
                vvv0.append(out['vvv'][0][j].cpu().detach().numpy().tolist())
                vvv1.append(out['vvv'][1][j].cpu().detach().numpy().tolist())
                count += 1
            if count > maxi:
                print("------------------------------------",i)
                break
        if count > maxi:
                print("------------------------------------",i)
                break            
            
        # print("grjyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
        # print(type(batch))
        
        
        # with torch.no_grad():
            # out= model(batch)
    
    
    
    
    
    
    V = Visualize(on= True, path = '/home/abr/Data/EMuRelPAFramework/data/vqa/coco/raw/')
    # a = [1,5,3,8,6]
    # b = np.array(a)

    # b = (b-min(b))/(max(b)-min(b))
    # print(str(b) +"\n")

    # print(str(v.sigmoid(a))+"\n")
    # print(str(v.sigmoid(b))+"\n")
    # import json
  
    # # Opening JSON file
    # with open('/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/representations_original.json') as json_file:
    #     data = json.load(json_file)
  

    # print (len(data))
    # for i in range(0, len(data)):
    #     item = data[i]
    #     # if 'time' in item['type']:
    #     if 'man' in item['answer'] or 'woman' in item['answer']:
    #     # if 'is the' in item['type'] and not ('what' in item['type']) and not ('or' in item['type']) and not ('where' in item['type']):
    #         if item['answer'] == item['answer']:
    #             items.append(item)
    #             vis_imgs.append(item['image_name'])
    #             vis_ques.append(item['question'])
    #             qq.append(item['q_reas'])
    #             q.append(item['q_agg'])
    #             mm.append(item['v_reas'])
    #             v.append(item['v_agg'])
    #             ans.append(item['answer'])
    #             pre.append(item['predicted'])
    #             count += 1
    #         if count > maxi:
    #             print("------------------------------------",i)
    #             break
        
    print(vis_ques)
    
    # vis_imgs.pop(1)
    # items.pop(1)

    # vis_ques.pop(1)
    # qq.pop(1)
    # q.pop(1)
    # mm.pop(1)
    # v.pop(1)
    # ans.pop(1)
    # pre.pop(1)
    #------------------------------------------------- arrange answers
    
    
    # vis_imgs = np.array(vis_imgs)
    # vis_ques = np.array(vis_ques)
    qq = np.array(qq)
    q = np.array(q)
    mm = np.array(mm)
    v = np.array(v)
    cq = np.array(cq)
    cv = np.array(cv)
    V.multi_image(vis_imgs)
    text = ""
    for i in range(len(vis_ques)):
        text += "\n" + str(i) + ")" + str(vis_ques[i])
        # print("::::::::::::::::::::::")
    seperator = V.seperator("-------q_txt")
    V.showText(text, seperator)

    V.cosine_similarity(qq, q, title = "Question", np_ = True)
    V.cosine_similarity(mm, v, title = "image", np_ = True)
    tx = "Answer:"
    for i in range(len(ans)):
        tx += "\n " + str(i) + ")" + ans[i]
    V.showText(tx)
    
    tx = "Predicted:"
    for i in range(len(ans)):
        tx += "\n " + str(i) + ")" + pre[i]
    V.showText(tx)
    
    tx = "Correct Prediction:"
    for i in range(len(ans)):
        c = False
        if pre[i] == ans[i]:
            c = True
        tx += "\n " + str(i) + ")" + str(c)
    V.showText(tx)
    
    print("a")
    # Opening JSON file
    import json
    with open('/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/'+file_name+'.json') as json_file:
        data = json.load(json_file)
    print("b")
    
 
  
    cell_q_1 = data[an1][1]['cellFinal_QF']
    cell_q_2 = data[an2][1]['cellFinal_QF']
    cell_v_1 = data[an1][0]['cellFinal_SF']
    cell_v_2 = data[an2][0]['cellFinal_SF']
    
    average_qq_1 = data[an1][1]['cellFinal_Q_RF']
    average_qq_2 = data[an2][1]['cellFinal_Q_RF']
    average_q_1 = data[an1][1]['agg_Q'] 
    average_q_2 = data[an2][1]['agg_Q'] 
    average_v_1 = data[an1][0]['agg_I']
    average_v_2 = data[an2][0]['agg_I']
    average_mm_1 = data[an1][0]['cellFinal_I_RF'] 
    average_mm_2 = data[an2][0]['cellFinal_I_RF'] 
    print(average_qq_1[4])
    print(average_qq_2[4])
    print(average_qq_1[23])
    print(average_qq_2[23])
    
    print("c")
    average_qq_1 = [average_qq_1] * (maxi +1)
    average_qq_2 = [average_qq_2] * (maxi +1)
    average_q_1 = [average_q_1] * (maxi +1)
    average_q_2 = [average_q_2] * (maxi +1)
    average_v_1 = [average_v_1] * (maxi +1)
    average_v_2 = [average_v_2] * (maxi +1)
    average_mm_1 = [average_mm_1] * (maxi +1)
    average_mm_2 = [average_mm_2] * (maxi +1)
    cell_q_1 = [cell_q_1] * (maxi +1)
    cell_q_2 = [cell_q_2] * (maxi +1)
    cell_v_1 = [cell_v_1] * (maxi +1)
    cell_v_2 = [cell_v_2] * (maxi +1)   
    print("d")
    average_qq = average_qq_1 + average_qq_2
    average_qq = np.array(average_qq)
    average_q = average_q_1 + average_q_2
    average_q = np.array(average_q)
    average_v = average_v_1 + average_v_2
    average_v = np.array(average_v)
    average_mm = average_mm_1 + average_mm_2
    average_mm = np.array(average_mm)
    cell_q = cell_q_1 + cell_q_2
    cell_q = np.array(cell_q)
    cell_v = cell_v_1 + cell_v_2
    cell_v = np.array(cell_v)   
    print("e")
    # dict_anws[key][1]['cellFinal_Q'] = dict_anws[key][1]['cellFinal_Q'].tolist()
    #     dict_anws[key][1]['cellFinal_QF'] = dict_anws[key][1]['cellFinal_QF'].tolist()
    #     dict_anws[key][1]['cellFinal_Q_RF'] = dict_anws[key][1]['cellFinal_Q_RF'].tolist()
    
    # V.cosine_similarity(qq, average_qq, title = "Question", np_ = True)
    print(qq.shape)
    print(average_qq.shape)
    
    V.cosine_similarity_reas_agg(q, average_q, "agg_question_average", np_ = True)
    V.cosine_similarity_reas_agg(qq, average_qq, "res_question_average", np_ = True)
    V.cosine_similarity_reas_agg(v, average_v, "agg_image_average", np_ = True)
    V.cosine_similarity_reas_agg(mm, average_mm, "res_image_average", np_ = True)
    # V.cosine_similarity_reas_agg(mm, mm, "res_image_average", np_ = True)
    # V.cosine_similarity_reas_agg(cq, cell_q, "cekk_Question_average", np_ = True)
    # V.cosine_similarity_reas_agg(cv, cell_v, "cell_image_average", np_ = True)
    # V.cosine_similarity_reas_agg(mm, v, "res_image", np_ = True)
    
    # V.distance(qq, q, title = "Question", np_ = True)
    # V.distance(mm, v, title = "image", np_ = True)




def calculate_average_similarities (batch_loader, file_name="averaged_dict"):
    
    
    import json
    with open('/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/'+file_name+'.json') as json_file:
        data = json.load(json_file)
    print("Dictionary Opened")
    
    # ---------------------------------------- For comparison
    with open('/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/murel_averaged_val_dictavg_vai_processed.json') as json_file:
        data2 = json.load(json_file)
    print("Dictionary Opened")    

    dict_occur = {}
    
    import torch.nn as nn


    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    from scipy import spatial


    count = {}
    countA = {}
    
    
    
    for i, batch in enumerate(batch_loader):
        if i > 3500:
            break
        out = m.forward2(batch)
        for j in range(len(batch['answer'])):
            # print(batch.keys())
            
                # print("gggggggggggggggggg", out['answers'][j])
            
            a = batch['answer'][j]
            countA[a] = 1
            # if a in data.keys() and a in data2.keys():
            if a in data.keys():
                average_qq = data[a][1]['cellFinal_Q_RF']
                average_mm = data[a][0]['cellFinal_I_RF'] 
            
            

                if a in dict_occur.keys():
                    # if dict_occur[a] > 5:
                    #     break
                    # else:
                    dict_occur[a] += 1
                else:
                    dict_occur[a] = 1
                #cityblock
                #cosine
                if a in data.keys():   
                    similarity_cosine_q = spatial.distance.cityblock(average_qq,  out['q_reas'][j].cpu().detach().numpy())
                    similarity_cosine_v = spatial.distance.cityblock(average_mm,  out['v_reas'][j].cpu().detach().numpy())
                    # print(similarity_cosine_q)
                    # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                    # if 'average_similarity' in data[a][0].keys() and 'variance_similarity' in data[a][1].keys():
                    try:    
                        data[a][0]['average_similarity'] = data[a][0]['average_similarity'] + similarity_cosine_v
                        data[a][1]['average_similarity'] = data[a][1]['average_similarity'] + similarity_cosine_q
                    except:
                        data[a][0]['average_similarity'] = similarity_cosine_v
                        data[a][1]['average_similarity'] = similarity_cosine_q
                else:
                    print("-=-=-=-=-=-%%+%+%+%%+%+%+%%+ not found in data", a)
            else:
                count[a] =0
                print("-------------- not found answer:  ", a)
                # print("-------------- number of not found keys ", count)
                
                
        print("-------------- Processed batch ", i)
        # if count > maxi:
        #     break
    
    
    for key, value in dict_occur.items():
        d = dict_occur[key]
        data[key][0]['average_similarity'] = data[key][0]['average_similarity'] / d
        data[key][1]['average_similarity'] = data[key][1]['average_similarity'] / d
        
        
    
    print("dumping file -----")
    data = np_to_list(data)
    with open("/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/"+ file_name +"avg_vai_processed.json", "w") as fp:
        json.dump(data, fp)     
        
    print("Dictionary closed ")   
        

    with open("/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/"+ file_name +"avg_vai_processed.json") as json_file:
        data = json.load(json_file)
    print("Dictionary opened again")
    
    
    average_qq_keys = 0 
    average_vv_keys = 0 
    number_of_keys = 0
    
    for key in (dict_occur.keys()):
        
        # print(data[key][1]['average_similarity'])
        # print("----------------------------------------------------------")
        average_qq_keys += data[key][1]['average_similarity']
        average_vv_keys += data[key][0]['average_similarity'] 
        number_of_keys += 1

    
    
    average_qq_keys = average_qq_keys / number_of_keys
    average_vv_keys = average_vv_keys / number_of_keys
    
    
    
    
    dict_occur = {}
    
    for i, batch in enumerate(batch_loader):
        if i > 3500:
            break
        out = m.forward2(batch)
        for j in range(len(batch['answer'])):
            # print(batch.keys())
            
                # print("gggggggggggggggggg", out['answers'][j])
            
            a = batch['answer'][j]
            # countA[a] = 1
            # if a in data.keys() and a in data2.keys():
            if a in data.keys():
                average_qq = data[a][1]['cellFinal_Q_RF']
                average_mm = data[a][0]['cellFinal_I_RF'] 
            
            

                if a in dict_occur.keys():
                    # if dict_occur[a] > 5:
                    #     break
                    # else:
                    dict_occur[a] += 1
                else:
                    dict_occur[a] = 1
                
                
            
                if a in data.keys():
                    
                    similarity_cosine_q = spatial.distance.cityblock(average_qq,  out['q_reas'][j].cpu().detach().numpy())
                    similarity_cosine_v = spatial.distance.cityblock(average_mm,  out['v_reas'][j].cpu().detach().numpy())
                    
                    similarity_cosine_q_var = np.exp(similarity_cosine_q - data[a][1]['average_similarity'])
                    similarity_cosine_v_var = np.exp(similarity_cosine_v - data[a][0]['average_similarity'])
                    
                    # if 'variance_similarity' in data[a][0].keys() and 'variance_similarity' in data[a][1].keys():
                    try:    
                        data[a][0]['variance_similarity'] = data[a][0]['variance_similarity'] + similarity_cosine_v_var
                        data[a][1]['variance_similarity'] = data[a][1]['variance_similarity'] + similarity_cosine_q_var
                    except:
                        data[a][0]['variance_similarity'] = similarity_cosine_v_var
                        data[a][1]['variance_similarity'] = similarity_cosine_q_var
                else:
                    print("-=-=-=-=-=-%%+%+%+%%+%+%+%%+ not found in data", a)
            else:
                # count[a] =0
                print("-------------- not found answer:  ", a)
                # print("-------------- number of not found keys ", count)
                
                
        print("-------------- Processed batch ", i)
        # if count > maxi:
        #     break
    import math
    for key, value in dict_occur.items():
        d = dict_occur[key]
        data[key][0]['variance_similarity'] = data[key][0]['variance_similarity'] / d
        data[key][1]['variance_similarity'] = data[key][1]['variance_similarity'] / d
        # try:
        print(data[key][0]['variance_similarity'])
        data[key][0]['standard_deviation'] = math.sqrt(data[key][0]['variance_similarity'])
        # except:
            # print("Error in standard_deviation")
        # try:
        print(data[key][0]['variance_similarity'])
        data[key][1]['standard_deviation'] = math.sqrt(data[key][1]['variance_similarity'])
        # except:
            # print("Error in standard_deviation")
            
    print("dumping file -----")
    data = np_to_list(data)
    with open("/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/"+ file_name +"avg_vai_processed.json", "w") as fp:
        json.dump(data, fp)     
        
    print("Dictionary closed ")   
        

    with open("/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/"+ file_name +"avg_vai_processed.json") as json_file:
        data = json.load(json_file)
    print("Dictionary opened again")    
    
    
    

    variance_qq_keys_avg = 0 
    variance_vv_keys_avg = 0 
    sd_qq_keys_avg = 0 
    sd_vv_keys_avg = 0 
    number_of_keys = 0
    
    for key in (dict_occur.keys()):
        

        
            
         variance_qq_keys_avg += data[key][1]['variance_similarity']
         variance_vv_keys_avg += data[key][0]['variance_similarity'] 
         try:
            sd_qq_keys_avg += data[key][1]['standard_deviation']
            sd_vv_keys_avg += data[key][0]['standard_deviation']
         except:
            sd_qq_keys_avg = 9
            sd_vv_keys_avg = 9
         number_of_keys += 1

    
    
    variance_qq_keys_avg = variance_qq_keys_avg / number_of_keys
    variance_vv_keys_avg = variance_vv_keys_avg / number_of_keys
    sd_qq_keys_avg = variance_qq_keys_avg / number_of_keys
    sd_vv_keys_avg = variance_vv_keys_avg / number_of_keys
    
    print("-------------------------------------------------------")
    print("average of all qq keys : ", average_qq_keys)
    print("average of all vv keys : ", average_vv_keys)
    print("variance of all qq keys : ", variance_qq_keys_avg)
    print("variance of all vv keys : ", variance_vv_keys_avg)
    print("standard deviation of all qq keys : ", sd_qq_keys_avg)
    print("standard deviation of all vv keys : ", sd_vv_keys_avg)
    print("number of all keys : ", number_of_keys)
    print("number of answers : ", len(countA.keys()))
    print("number of missing answers : ", len(count.keys()))
    
  

    tx = ""
    tx += "average of all qq keys : " + str (average_qq_keys) + "\n"
    tx += "average of all vv keys : " + str (average_vv_keys)+ "\n"
    tx +="variance of all qq keys : "+ str (variance_qq_keys_avg)+ "\n"
    tx +="variance of all vv keys : "+ str (variance_vv_keys_avg)+ "\n"
    tx +="standard deviation of all qq keys : "+ str (sd_qq_keys_avg)+ "\n"
    tx +="standard deviation of all vv keys : "+ str (sd_vv_keys_avg)+ "\n"
    tx +="number of all keys : "+ str ( number_of_keys)+ "\n"
    tx +="number of answers : "+ str ( len(countA.keys()))+ "\n"
    tx +="number of missing answers : "+ str ( len(count.keys()))+ "\n"
    
    V = Visualize(on= True, path = '/home/abr/Data/EMuRelPAFramework/data/vqa/coco/raw/')
    V.showText(tx)
  
   


def np_to_list(dict_anws):
    print("np to list -----")
    for key, value in dict_anws.items():
        for key2, value in dict_anws[key][0].items():
            try:
                dict_anws[key][0][key2] = dict_anws[key][0][key2].tolist()
            except:
                print("int object")#, dict_anws[key][0][key2])
        for key2, value in dict_anws[key][1].items():
            try:
                dict_anws[key][1][key2] = dict_anws[key][1][key2].tolist()
            except:
                print("int object")#, dict_anws[key][1][key2])
        # print(key)
        # dict_anws[key][0]['agg_I'] = dict_anws[key][0]['agg_I'].tolist()
        # dict_anws[key][0]['cellFinal_I'] = dict_anws[key][0]['cellFinal_I'].tolist()
        # dict_anws[key][0]['cellFinal_SF'] = dict_anws[key][0]['cellFinal_SF'].tolist()
        # dict_anws[key][0]['cellFinal_I_RF'] = dict_anws[key][0]['cellFinal_I_RF'].tolist()
        # dict_anws[key][1]['agg_Q'] = dict_anws[key][1]['agg_Q'].tolist()
        # dict_anws[key][1]['cellFinal_Q'] = dict_anws[key][1]['cellFinal_Q'].tolist()
        # dict_anws[key][1]['cellFinal_QF'] = dict_anws[key][1]['cellFinal_QF'].tolist()
        # dict_anws[key][1]['cellFinal_Q_RF'] = dict_anws[key][1]['cellFinal_Q_RF'].tolist()
    return dict_anws



   


def np_to_list(dict_anws):
    print("np to list -----")
    for key, value in dict_anws.items():
        for key2, value in dict_anws[key][0].items():
            try:
                dict_anws[key][0][key2] = dict_anws[key][0][key2].tolist()
            except:
                print("int object")#, dict_anws[key][0][key2])
        for key2, value in dict_anws[key][1].items():
            try:
                dict_anws[key][1][key2] = dict_anws[key][1][key2].tolist()
            except:
                print("int object")#, dict_anws[key][1][key2])
        # print(key)
        # dict_anws[key][0]['agg_I'] = dict_anws[key][0]['agg_I'].tolist()
        # dict_anws[key][0]['cellFinal_I'] = dict_anws[key][0]['cellFinal_I'].tolist()
        # dict_anws[key][0]['cellFinal_SF'] = dict_anws[key][0]['cellFinal_SF'].tolist()
        # dict_anws[key][0]['cellFinal_I_RF'] = dict_anws[key][0]['cellFinal_I_RF'].tolist()
        # dict_anws[key][1]['agg_Q'] = dict_anws[key][1]['agg_Q'].tolist()
        # dict_anws[key][1]['cellFinal_Q'] = dict_anws[key][1]['cellFinal_Q'].tolist()
        # dict_anws[key][1]['cellFinal_QF'] = dict_anws[key][1]['cellFinal_QF'].tolist()
        # dict_anws[key][1]['cellFinal_Q_RF'] = dict_anws[key][1]['cellFinal_Q_RF'].tolist()
    return dict_anws

    
def list_to_np(dict_anws):
    print("list to np -----")
    for key, value in dict_anws.items():
       
        for key2, value in dict_anws[key][0].items():
            try:
                dict_anws[key][0][key2] = np.array(dict_anws[key][0][key2])
            except:
                print("int object")#, dict_anws[key][0][key2])
        for key2, value in dict_anws[key][1].items():
            try:
                dict_anws[key][1][key2] = np.array(dict_anws[key][1][key2])
            except:
                print("int object")#, dict_anws[key][1][key2])
        # print(key)
        # dict_anws[key][0]['agg_I'] = np.array(dict_anws[key][0]['agg_I'])
        # dict_anws[key][0]['cellFinal_I'] = np.array(dict_anws[key][0]['cellFinal_I'])
        # dict_anws[key][0]['cellFinal_SF'] = np.array(dict_anws[key][0]['cellFinal_SF'])
        # dict_anws[key][0]['cellFinal_I_RF'] = np.array(dict_anws[key][0]['cellFinal_I_RF'])
        # dict_anws[key][1]['agg_Q'] = np.array(dict_anws[key][1]['agg_Q'])
        # dict_anws[key][1]['cellFinal_Q'] = np.array(dict_anws[key][1]['cellFinal_Q'])
        # dict_anws[key][1]['cellFinal_QF'] = np.array(dict_anws[key][1]['cellFinal_QF'])
        # dict_anws[key][1]['cellFinal_Q_RF'] = np.array(dict_anws[key][1]['cellFinal_Q_RF'])
    return dict_anws

def prepare_dict (batch_loader, file_name='averaged_dict'):
    import numpy as np
    defaultList_agg_I = np.array([0] *6576)
    defaultList_agg_Q = np.array([0] *4800)
    
    
    
   
    
    from collections import defaultdict
    
    # dict_anws_I = defaultdict(lambda: defaultList_agg_I, dict_anws_I)
    # dict_anws_Q = defaultdict(lambda: defaultList_agg_Q, dict_anws_Q)
    dict_anws = {}
    dict_occur = {}
    dict_anws_I = {'agg_I': defaultList_agg_I, 
                'cellFinal_I': defaultList_agg_I, 
                 'cellFinal_SF': defaultList_agg_I,
                 'cellFinal_I_RF': defaultList_agg_I
                 
                 }
    dict_anws_Q = {'agg_Q': defaultList_agg_Q,
                 'cellFinal_Q': defaultList_agg_Q, 
                  'cellFinal_QF': defaultList_agg_Q,
                 'cellFinal_Q_RF': defaultList_agg_Q
                 }   
    
    # dict_anws = defaultdict(lambda: [dict_anws_I, dict_anws_Q], dict_anws)
    
    # dict_occur = defaultdict(lambda: 0, dict_occur)
    
    dict_anws = np_to_list(dict_anws)
    import json
    print("dumping file -----")
    with open("/home/abr/Data/MuHiPAFramework/MuHiPA/models/networks/"+ file_name +".json", "w") as fp:
        json.dump(dict_anws, fp)
    
    print("opening file -----")
    with open('/home/abr/Data/MuHiPAFramework/MuHiPA/models/networks/'+file_name+'.json') as json_file:
        dict_anws = json.load(json_file)
    dict_anws = list_to_np(dict_anws)
    maxi = 10000
    count = 0
    lasti = 0
    for i, batch in enumerate(batch_loader):
        if i > 3500:
            break
        out = m.forward2(batch)
        for j in range(len(batch['answer'])):
            # print(batch.keys())
            if out['answers'][j] == batch['answer'][j]:
                # print("gggggggggggggggggg", out['answers'][j])

                a = batch['answer'][j]

                if a in dict_occur.keys():
                    # if dict_occur[a] > 5:
                    #     break
                    # else:
                    dict_occur[a] = dict_occur[a] + 1
                else:
                    dict_occur[a] = 1
                
                
                
                if a in dict_anws.keys():
                    dict_anws[a][0]['agg_I'] = dict_anws[a][0]['agg_I'] + out['v_agg'][j].cpu().detach().numpy()
                    try:
                        dict_anws[a][0]['cellFinal_I'] = dict_anws[a][0]['cellFinal_I'] + out['cell'][2][1][j].cpu().detach().numpy()
                        dict_anws[a][0]['cellFinal_SF'] = dict_anws[a][0]['cellFinal_SF'] + out['cell'][2][0][j].cpu().detach().numpy()
                    except:
                        print("no cell")
                    dict_anws[a][0]['cellFinal_I_RF'] = dict_anws[a][0]['cellFinal_I_RF'] + out['v_reas'][j].cpu().detach().numpy()
                    dict_anws[a][1]['agg_Q'] = dict_anws[a][1]['agg_Q'] + out['q_agg'][j].cpu().detach().numpy()
                    try:
                        dict_anws[a][1]['cellFinal_Q'] = dict_anws[a][1]['cellFinal_Q'] + out['cell'][2][3][j].cpu().detach().numpy()
                        dict_anws[a][1]['cellFinal_QF'] = dict_anws[a][1]['cellFinal_QF'] + out['cell'][2][2][j].cpu().detach().numpy()
                    except:
                        print("no cell")
                    dict_anws[a][1]['cellFinal_Q_RF'] = dict_anws[a][1]['cellFinal_Q_RF'] + out['q_reas'][j].cpu().detach().numpy()
                else:
                    
                    defaultList_agg_I_1 = out['v_agg'][j].cpu().detach().numpy()
                    defaultList_agg_Q_1 = out['q_agg'][j].cpu().detach().numpy()
                    
                    try:
                        defaultList_agg_I_2 = out['cell'][2][1][j].cpu().detach().numpy()
                        defaultList_agg_Q_2 = out['cell'][2][3][j].cpu().detach().numpy()
                    
                        defaultList_agg_I_3 = out['cell'][2][0][j].cpu().detach().numpy()
                        defaultList_agg_Q_3 = out['cell'][2][2][j].cpu().detach().numpy()
                    except:
                        print("No Cell")
                        defaultList_agg_I_2 = out['v_agg'][j].cpu().detach().numpy()
                        defaultList_agg_Q_2 =out['v_agg'][j].cpu().detach().numpy()
                    
                        defaultList_agg_I_3 = out['v_agg'][j].cpu().detach().numpy()
                        defaultList_agg_Q_3 = out['v_agg'][j].cpu().detach().numpy()
                        
                    defaultList_agg_I_4 = out['v_reas'][j].cpu().detach().numpy()
                    defaultList_agg_Q_4 = out['q_reas'][j].cpu().detach().numpy()
                    
                    dict_anws_I = {'agg_I': defaultList_agg_I_1, 
                'cellFinal_I': defaultList_agg_I_2,
                 'cellFinal_SF': defaultList_agg_I_3,
                 'cellFinal_I_RF': defaultList_agg_I_4
                 
                 }
                    dict_anws_Q = {'agg_Q': defaultList_agg_Q_1,
                 'cellFinal_Q': defaultList_agg_Q_2, 
                  'cellFinal_QF': defaultList_agg_Q_3,
                 'cellFinal_Q_RF': defaultList_agg_Q_4
                 }   
                    dict_anws[a] = [dict_anws_I, dict_anws_Q]
                count = count + 1
                
                
                
            # if count > maxi:
                # count = 0
                # print("dumping file -----")
                # dict_anws = np_to_list(dict_anws)
                # with open("/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/"+ file_name +str(i)+".json", "w") as fp:
                #     json.dump(dict_anws, fp)
                    
        
                # print("opening file -----")
                # with open('/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/'+file_name+ str(i)+'.json') as json_file:
                #     dict_anws = json.load(json_file)
                # dict_anws = list_to_np(dict_anws)   
                # lasti = i
                # break
        
        print("-------------- Processed batch ", i)
        # if count > maxi:
        #     break
        
        
    # print("dumping file -----")
    # dict_anws = np_to_list(dict_anws)
    # with open("/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/"+ file_name +str(0)+".json", "w") as fp:
    #     json.dump(dict_anws, fp)
        
        
    # print("opening file -----")
    # with open('/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/'+file_name+ str(0)+'.json') as json_file:
    #     dict_anws = json.load(json_file)
    # dict_anws = list_to_np(dict_anws)        
    
    for key, value in dict_anws.items():
        d = dict_occur[key]
        dict_anws[key][0]['agg_I'] = dict_anws[key][0]['agg_I'] / d
        dict_anws[key][0]['cellFinal_I'] = dict_anws[key][0]['cellFinal_I'] / d
        dict_anws[key][0]['cellFinal_SF'] = dict_anws[key][0]['cellFinal_SF'] / d
        dict_anws[key][0]['cellFinal_I_RF'] = dict_anws[key][0]['cellFinal_I_RF'] / d
        dict_anws[key][1]['agg_Q'] = dict_anws[key][1]['agg_Q'] / d
        dict_anws[key][1]['cellFinal_Q'] = dict_anws[key][1]['cellFinal_Q'] / d
        dict_anws[key][1]['cellFinal_QF'] = dict_anws[key][1]['cellFinal_QF'] / d
        dict_anws[key][1]['cellFinal_Q_RF'] = dict_anws[key][1]['cellFinal_Q_RF'] / d
    
    print("dumping file -----")
    dict_anws = np_to_list(dict_anws)
    with open("/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/"+ file_name +".json", "w") as fp:
        json.dump(dict_anws, fp)            
    
    # for key, value in dict_anws.items():
    #     print(key)
    #     dict_anws[key][0]['agg_I'] = dict_anws[key][0]['agg_I'].tolist()
    #     dict_anws[key][0]['cellFinal_I'] = dict_anws[key][0]['cellFinal_I'].tolist()
    #     dict_anws[key][0]['cellFinal_SF'] = dict_anws[key][0]['cellFinal_SF'].tolist()
    #     dict_anws[key][0]['cellFinal_I_RF'] = dict_anws[key][0]['cellFinal_I_RF'].tolist()
    #     dict_anws[key][1]['agg_Q'] = dict_anws[key][1]['agg_Q'].tolist()
    #     dict_anws[key][1]['cellFinal_Q'] = dict_anws[key][1]['cellFinal_Q'].tolist()
    #     dict_anws[key][1]['cellFinal_QF'] = dict_anws[key][1]['cellFinal_QF'].tolist()
    #     dict_anws[key][1]['cellFinal_Q_RF'] = dict_anws[key][1]['cellFinal_Q_RF'].tolist()
        
    
    
    # import json
    # with open("/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/"+ file_name +".json", "w") as fp:
    #     json.dump(dict_anws, fp)
        
    return 0

if __name__ == '__main__': 
    
    

    ooo = '-o /home/abr/Data/EMuRelPAFramework/logs/vqa2/server/fromServer/newModel/emurelpa_trainval2/options.yaml --dataset.batch_size 10 --exp.resume best_eval_epoch.accuracy_top1 --dataset.train_split --dataset.eval_split val --misc.logs_name test --exp.dir /home/abr/Data/EMuRelPAFramework/logs/vqa2/server/fromServer/newModel/emurelpa_trainval2 --dataset.dir /home/abr/Data/EMuRelPAFramework/data/vqa/vqa2 --model.network.txt_enc.dir_st /home/abr/Data/EMuRelPAFramework/data/skip-thoughts --dataset.dir_rcnn /home/abr/Data/EMuRelPAFramework/data/vqa/data/vqa/coco/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36'
    # path_opts=""
    # print(path_opts)
    from bootstrap.lib.options import Options
    Options(None)
    from bootstrap.lib import utils
    utils.set_random_seed(Options()['misc']['seed'])
    
    import torch
    
    import torch.backends.cudnn as cudnn
    if torch.cuda.is_available():
            cudnn.benchmark = True
    
    import bootstrap.engines as engines
    engine = engines.factory()
    import bootstrap.datasets as datasets
    engine.dataset = datasets.factory(engine)
    import bootstrap.models as models
    engine.model = models.factory(engine)
    m = engine.model
    import bootstrap.optimizers as optimizers
    engine.optimizer = optimizers.factory(m, engine)
    import bootstrap.views as views
    engine.view = views.factory(engine)
    
    if Options()['exp']['resume']:
            engine.resume(None if Options().get('misc.cuda', False) else 'cpu')
    
    utils.set_random_seed(Options()['misc']['seed'] + engine.epoch)
    
    m.eval()
    # batch_loader = engine.dataset['eval'].make_batch_loader()
    # engine.hook('train_on_start')
    print("-------bbbbb----", engine.dataset.keys())
    batch_loader = engine.dataset["eval"].make_batch_loader()
    # mode='train'
    # engine.hook(f'{mode}_on_start_epoch')
    # self.hook('{}_on_start_epoch'.format(mode))
    
    engine.hook('{}_on_start_batch'.format('eval'))
    
    
    print("-------bbbbb----", len(batch_loader))
    default = "averaged_dict00"
    
    prepare_dict (batch_loader, file_name="murel_averaged_val_dict")
    calculate_average_similarities (batch_loader, file_name="murel_averaged_val_dict")
    # distance_cosine (batch_loader, file_name="averaged_val_dict")
    
    
    


    # from pthflops import count_ops

    # Create a network and a corresponding input
    # device = 'cuda:0'

    # inp = torch.rand(1,3,224,224).to(device)

    # Count the number of FLOPs
    # count_ops(m, next(iter(batch_loader)))
    
    # import json
    # with open('/home/abr/Data/EMuRelPAFramework/EMuRelPA/models/networks/'+default+'.json') as json_file:
    #     data = json.load(json_file)
    # print(data['lady'][0]['agg_I'])
    
    
    
   
