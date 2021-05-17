import math
import copy
import sys

def load_corpus(path):
    f1 = open(path)
    taggedSentences = []

    curLine = f1.readline()
    while curLine:
        tags = []

        # Convert line to list of tagged words
        words = curLine.split(" ")
        for word in words:
            tag = word.split("=")
            tags.append((tag[0],tag[1]))
        
        # remove new line charachter and add to tagged
        tags[len(words)-1] = (tags[len(words)-1][0],tags[len(words)-1][1][:-1])
        taggedSentences.append(tags)

        curLine = f1.readline()
    
    return taggedSentences


class Tagger(object):

    def __init__(self, sentences):
        self.pos = ('ADV', 'NOUN', 'ADP','PRT','DET','.','PRON','VERB','X','NUM','CONJ','ADJ')
        smoothing = 1e-5

        #number of times a self.pos tag is the first
        tagOccurance = {p2:0 for p2 in self.pos}

        self.tagProbs = copy.deepcopy(tagOccurance)
        self.transProbs = {p1:copy.deepcopy(tagOccurance) for p1 in self.pos}
        self.emProb = {p2:{} for p2 in self.pos}

        print("\n\n")
        t = 0
        for sentence in sentences:
            print("\rTraining: ",int((t/57340)*100),"%",end='')
            t+=1

            # Load tagProbs with number of occurances of first word in sentance
            self.tagProbs[sentence[0][1]]+=1
            
            for i in range(len(sentence)):
                # prepare tagOccurance transProbs which will be used to calculate transProbs
                tagOccurance[sentence[i][1]]+=1

                if i<(len(sentence)-1):
                    self.transProbs[sentence[i][1]][sentence[i+1][1]]+=1
                
                if  not self.emProb[sentence[i][1]].get(sentence[i][0]):
                    self.emProb[sentence[i][1]][sentence[i][0]]=1
                else:
                    self.emProb[sentence[i][1]][sentence[i][0]]+=1
        
        for tag in self.transProbs:
            for tag2 in self.transProbs[tag]:
                self.transProbs[tag][tag2] = (smoothing+self.transProbs[tag][tag2])/float((smoothing*(len(self.pos)+1)+tagOccurance[tag]-self.transProbs[tag][tag2]))

        # Convert number of occurances in tagProbs to probability of word
        for tag in self.tagProbs:
            self.tagProbs[tag] = self.tagProbs[tag]/float(len(sentences))
        
        numWords = 0

        for tag in self.emProb :
            ln = len(self.emProb[tag].keys())+1
            for word in self.emProb[tag]:
                # self.emProb[tag][word]=self.emProb[tag][word]/tagOccurance[tag]
                self.emProb[tag][word]=(smoothing+self.emProb[tag][word])/(smoothing*(ln)+numWords)
                # self.emProb[tag][word]=math.log(float(self.emProb[tag][word] + smoothing) / (numWords + len(self.emProb[tag].keys())*smoothing))
            self.emProb[tag]["<UNK>"]=(smoothing)/(smoothing*(ln)+numWords)
        
        print("\rTraining: ",100,"%")

    def most_probable_tags(self, tokens):
        ans = []
        # maxProb = 0
        
        for token in tokens:
            maxProb = -float('inf')
            for tag in self.emProb.keys():
                if token not in self.emProb[tag]:
                    if self.emProb[tag]["<UNK>"]>maxProb:
                        maxProb = self.emProb[tag]["<UNK>"]
                        maxT="X"
                else:
                    if self.emProb[tag][token]>maxProb:
                        maxProb = self.emProb[tag][token]
                        maxT=tag
                    # maxT= tag
            ans.append(maxT)
        return ans


    def viterbi_tags(self, tokens):
        z=[]
        bk=[]
        i = 0
        maxValue = -float('inf')
        ans = []

        #Initialize z and bk
        for token in tokens:
            zp = []
            bkp = []
            for p in range(12):
                bkp.append(0)
                zp.append(float(0))
            z.append(zp)
            bk.append(bkp)

        itt = 0
        for tag in self.pos:
            if not self.emProb[tag].get(tokens[0]):
                emmission = self.emProb[tag]["<UNK>"]
            else:
                emmission = self.emProb[tag][tokens[0]] 
            z[0][itt] = self.tagProbs[tag]*emmission
            itt+=1
        
        for t in range(len(tokens)-1):
            for t2 in range(len(self.pos)):
                for t1 in range(len(self.pos)):
                    if z[t][t1]*self.transProbs[self.pos[t1]][self.pos[t2]] > maxValue:
                        maxValue = z[t][t1]*self.transProbs[self.pos[t1]][self.pos[t2]]
                        i = t1
                bk[t+1][t2] = i

                #set emmission
                if not self.emProb[self.pos[t2]].get(tokens[t+1]):
                    emmission = self.emProb[self.pos[t2]]["<UNK>"]
                else:
                    emmission = self.emProb[self.pos[t2]][tokens[t+1]] 
                z[t+1][t2] = maxValue*emmission
                
                #reset
                i = 0
                maxValue = -float('inf')
        
        for p in range(len(self.pos)):
            if z[-1][p] > maxValue:
                maxValue = z[-1][p]
                i = p
        # prev = i
        ans.insert(0, self.pos[i])

        prev = i
        for t in range(len(tokens)-1):
            prev = bk[len(tokens)-1-t][i]
            ans.insert(0,self.pos[prev])
        
        posColors = {'NOUN':"\033[95m", 'VERB':"\033[94m", 'ADV':"\033[93m", 'ADP':"\033[92m",'PRT':"\033[0m",'DET':"\033[90m",'.':"\033[97m",'PRON':"\033[1m\033[95m",'X':"\033[0;37;41m",'NUM':"",'CONJ':"\033[96m",'ADJ':"\033[91m"}
        posNames = {'NOUN':"Noun", 'VERB':"Verb", 'ADV':"Adverb", 'ADP':"Preposition or Postposition",'PRT':"Particle",'DET':"Determiner or Article",'.':"Punctuation",'PRON':"Pronoun",'X':"Unknown",'NUM':"",'CONJ':"\033[96m",'ADJ':"\033[91m"}

        print("\n\n\nLegend: ", end='')
        for i in posColors:
            print(posColors[i],posNames[i],"\033[0m ", end='')
        print("\n\n\n")

        result = ""
        for i in range(len(tokens)):
            result+=posColors[ans[i]]+tokens[i]+"\033[0m "
        result+="\n\n\n"
        return result

# Prep training data
c = load_corpus("brown-corpus.txt")

# Train model on training data
g = Tagger(c)

# Resolve queries
done=False
print("\n\n------------------------------------Type exit() to quit------------------------------------")
while not done:
    inp = input("\n\nType a sentence: ").split()
    if inp[0]=="exit()":
        done=True
    else:
        print(g.viterbi_tags(inp))
