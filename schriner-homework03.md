## PART 1: Data Exploration

To understand inflection in Polish, I started with Dąbrowski (2001)[^1]. Discussing the concepts of connection weights
and dual mechanism theory (default status does not depend on frequency), the author builds the case that English, 
in its simplicity "makes it unrepresentative."  The author builds the case, through English, then German, that ideally
we'd consider data from a number of languages.
In Polish, there are irregular classes that share particular inflections or even sets of inflections; describing
the system of the Polish genitive, one can see that it's.. complex.

**Some points:**
Gender can usually be predicted by the nomitive ending<br />
Some nouns belong to classes (e.g. tools and body parts) that require a certain ending (in this case -a)<br />
There is a lot of variation: (e.g. MASC nouns that take -a in the singular can take -ow or -i/-y in the plural)<br />
Stem changes (both regular and irregular, I believe) will make correct prediction via Fairseq very difficult<br />

I attempted to translate the words using googletrans but found that it's buggy at the moment.<br />
So I translated a small portion of the words to look at noun properties/classes.<br />

*Some generalizations:*
Feminine nouns that end in -a often have a genitive form of -y and seemingly to a lesser extent -i

- fatwa	fatwy	FEM	(fascist)
- fontanna	fontanny	FEM	(phoneme)
- fotosynteza	fotosyntezy	FEM	(moat)

- felga	felgi	FEM	(counterfeiter)
- fotka	fotki	FEM	(piano)

Inanimate masc. nouns that end in -n tend to add -u for the genitive.
So rather than replacing the -a with a -y like above, these tack on a -u, seemingly regularly:

- fosforan	fosforanu	MASC;INAN	(trick)
- fort	fortu	MASC;INAN	(phonetics)
- fragment	fragmentu	MASC;INAN	(photo)

Animate masc. nouns that end in -a tend to be replaced by a -y:<br />
a > y / _#

- fizjoterapeuta	fizjoterapeuty	MASC;HUM	(violet)
- flecista	flecisty	MASC;HUM	(physics)

Animate masc. nouns that end in a consonant, tend to get an -a:

- filozof	filozofa	MASC;HUM	(library)
- fizyk	fizyka	MASC;HUM	(business)

*Neuter nouns occur 645/6653 = 10.3%*

Ending -o changes to -a in the genitive (this seems like the majority of cases and is regular):

- drzewo	drzewa	NEUT	(tree)
- gówienko	gówienka	NEUT	(barnyard)
- gówno	gówna	NEUT	(shit)
- hasło	hasła	NEUT	(password)

Ending -e in neuters change to -a:

- głosowanie	głosowania	NEUT	(vote)
- istnienie	istnienia	NEUT	(existence)

Ending -ę in neuters get-cia added and it seems regular:

- jagnię	jagnięcia	NEUT	(lamb)
- kaczę	kaczęcia	NEUT	(duckling)
- lisię	lisięcia	NEUT	(fox)
Semantically, cute little animals

*Some generalizations about the Polish genitive system from Dąbrowski (2001):*

-a is the most frequent ending in masculin Polish nouns "accounting for 70-80% of masculine types as well as tokens."<br />
There is "no single ending applicable in all circumstances"<br />
Children should overgeneralize an ending, thinking it's the default.<br />
The exact opposite was true from the Szuman/Smoczyńska data-- overgeneralization was rare and children learn the
endings for the appropriate genders very early in development.<br />
"The distribution of -a and -u within the masculine declension causes more problems because it is largely arbitrary"<br />

That immediately makes me think that testing and training will have low/zero word error rates for certain nouns 
(neuters) but will predict poorly for some masculine nouns.

[^1]: Dąbrowska, E. (2001). Learning a morphological system without a default: 
    The Polish genitive. Journal of Child Language, 28(03). https://doi.org/10.1017/S0305000901004767
    
    
## PART 2: Splitting the Data

#Randomizing the data:
I've used shuf before and I think it's a fine solution, but if I need to code this in python, that's fine too.

`shuf -o rand_pol.tsv pol.tsv`

I used a Gorman & Lee script for splitting the data 80/10/10:<br />
https://github.com/CUNY-CL/wikipron-modeling/blob/master/scripts/split.py
```
python split.py --seed 145 \  
    --input_path rand_pol.tsv \ 
    --train_path pol_train.tsv \ 
    --dev_path pol_dev.tsv \
    --test_path pol_test.tsv
INFO: Total set:	6,653 lines
INFO: Train set:	5,322 lines
INFO: Development set:	665 lines
INFO: Test set:		666 lines
```
<p>
The output files:<br />
17K Nov  8 13:51 pol_dev.tsv<br />
16K Nov  8 13:51 pol_test.tsv<br />
129K Nov  8 13:51 pol_train.tsv<br />`

## PART 3: Preprocessing

n.b. code adapted from Kyle Gorman's preparation code at <br />
https://github.com/language-technology-GC/HW2-solution/blob/master/prepare.py<br />

```   
import csv
import contextlib

TRAIN = "pol_train.tsv"
TRAIN_N = "train.pol.nomsg"
TRAIN_G = "train.pol.gensg"

DEV = "pol_dev.tsv"
DEV_N = "dev.pol.nomsg"
DEV_G = "dev.pol.gensg"

TEST = "pol_test.tsv"
TEST_N = "test.pol.nomsg"
TEST_G = "test.pol.gensg"

def main() -> None:
    # Processes training data.
    with contextlib.ExitStack() as stack:
        source = csv.reader(
            stack.enter_context(open(TRAIN, "r")), delimiter="\t")
        n = stack.enter_context(open(TRAIN_N, "w"))
        g = stack.enter_context(open(TRAIN_G, "w"))
        for nom, gen, _ in source:
            print(" ".join(nom), file=n)
            print(" ".join(gen), file=g)
    # Processes development data.
    with contextlib.ExitStack() as stack:
        source = csv.reader(
            stack.enter_context(open(DEV, "r")), delimiter="\t")
        n = stack.enter_context(open(DEV_N, "w"))
        g = stack.enter_context(open(DEV_G, "w"))
        for nom, gen, _ in source:
            print(" ".join(nom), file=n)
            print(" ".join(gen), file=g)
    # Processes test data.
    with contextlib.ExitStack() as stack:
        source = csv.reader(
            stack.enter_context(open(TEST, "r")), delimiter="\t")
        n = stack.enter_context(open(TEST_N, "w"))
        g = stack.enter_context(open(TEST_G, "w"))
        for nom, gen, _ in source:
            print(" ".join(nom), file=n)
            print(" ".join(gen), file=g)


if __name__ == "__main__":
    main()
```
    
Output files:
11K Nov  8 14:00 dev.pol.gensg<br />
9.5K Nov  8 14:00 dev.pol.nomsg<br />
11K Nov  8 14:00 test.pol.gensg<br />
9.6K Nov  8 14:00 test.pol.nomsg<br />
82K Nov  8 14:00 train.pol.gensg<br />
77K Nov  8 14:00 train.pol.nomsg<br />

Manually inspected to check for spacing and did the following to check the number of rows:
```
wc -l dev*
  665 dev.pol.gensg
  665 dev.pol.nomsg
 1330 total
 
wc -l test*
  666 test.pol.gensg
  666 test.pol.nomsg
 1332 total
 
wc -l train*
  5322 train.pol.gensg
  5322 train.pol.nomsg
 10644 total
```
    
Added those totals, divided by 2 = 6653, the correct number of rows.<br />
*All data accounted for, randomized, split, and preprocessed for Fairseq.*<br />
Moved all older files not to be used in fairseq-preprocess to a new folder in root.<br />

```
fairseq-preprocess \
    --source-lang pol.nomsg \
    --target-lang pol.gensg \
    --trainpref train \
    --validpref dev \
    --testpref test \
    --tokenizer space \
    --thresholdsrc 2 \
    --thresholdtgt 2
    
Output:
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | Namespace(align_suffix=None, alignfile=None, all_gather_list_size=16384, bf16=False, bpe=None, checkpoint_shard_count=1, checkpoint_suffix='', cpu=False, criterion='cross_entropy', dataset_impl='mmap', destdir='data-bin', empty_cache_freq=0, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, joined_dictionary=False, log_format=None, log_interval=100, lr_scheduler='fixed', memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, model_parallel_size=1, no_progress_bar=False, nwordssrc=-1, nwordstgt=-1, only_source=False, optimizer=None, padding_factor=8, profile=False, quantization_config_path=None, scoring='bleu', seed=1, source_lang='pol.gensg', srcdict=None, target_lang='pol.nomsg', task='translation', tensorboard_logdir=None, testpref='test', tgtdict=None, threshold_loss_scale=None, thresholdsrc=2, thresholdtgt=2, tokenizer='space', tpu=False, trainpref='train', user_dir=None, validpref='dev', workers=1)
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.gensg] Dictionary: 40 types
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.gensg] train.pol.gensg: 5322 sents, 45952 tokens, 0.00435% replaced by <unk>
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.gensg] Dictionary: 40 types
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.gensg] dev.pol.gensg: 665 sents, 5735 tokens, 0.0% replaced by <unk>
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.gensg] Dictionary: 40 types
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.gensg] test.pol.gensg: 666 sents, 5713 tokens, 0.0% replaced by <unk>
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.nomsg] Dictionary: 40 types
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.nomsg] train.pol.nomsg: 5322 sents, 43412 tokens, 0.00921% replaced by <unk>
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.nomsg] Dictionary: 40 types
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.nomsg] dev.pol.nomsg: 665 sents, 5397 tokens, 0.0185% replaced by <unk>
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.nomsg] Dictionary: 40 types
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | [pol.nomsg] test.pol.nomsg: 666 sents, 5444 tokens, 0.0% replaced by <unk>
2021-11-08 14:16:30 | INFO | fairseq_cli.preprocess | Wrote preprocessed data to data-bin
```

## PART 4: LSTM training, Part 1
```
fairseq-train \
    data-bin \
    --source-lang pol.nomsg \
    --target-lang pol.gensg \
    --seed 445 \
    --arch lstm \
    --encoder-bidirectional \
    --dropout .2 \
    --encoder-embed-dim 128 \
    --decoder-embed-dim 128 \
    --decoder-out-embed-dim 128 \
    --encoder-hidden-size 512 \
    --decoder-hidden-size 512 \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing .1 \
    --optimizer adam \
    --lr .001 \
    --clip-norm 1 \
    --batch-size 128 \
    --max-update 4000 \
    --no-epoch-checkpoints \
    --save-dir extra
    
Output files: 
86M Nov  8 14:30 checkpoint_best.pt
86M Nov  8 14:50 checkpoint_last.pt
```
    
Evaluation of Dev and Test 
```
fairseq-generate \
    data-bin \
    --source-lang pol.nomsg \
    --target-lang pol.gensg \
    --path part4/checkpoint_best.pt \
    --gen-subset valid \
    --beam 8 \
    > predictions-dev.txt
```
n.b.Using wer.py from Gorman at: https://github.com/language-technology-GC/HW2-solution/blob/master/wer.py<br />
**Development set WER: 14.74**
```
fairseq-generate \
    data-bin \
    --source-lang pol.nomsg \
    --target-lang pol.gensg \
    --path part4/checkpoint_best.pt \
    --gen-subset test \
    --beam 8 \
    > predictions-test.txt
```
**Test set WER: 15.47**

## PART 5: LSTM training, Part 2

### 5a: Doubled the d/encoder hidden layer sizes, changed seed<br />
Each epoch is noticably much slower - this took > 2 hours<br />
Naïvely seeing if it's like a video encoder: slower, more thorough processing makes for better quality<br />
Clearly not in this case, with these parameters<br />
```
fairseq-train \
    data-bin \
    --source-lang pol.nomsg \
    --target-lang pol.gensg \
    --seed 288 \
    --arch lstm \
    --encoder-bidirectional \
    --dropout .2 \
    --encoder-embed-dim 128 \
    --decoder-embed-dim 128 \
    --decoder-out-embed-dim 128 \
    --encoder-hidden-size 1024 \
    --decoder-hidden-size 1024 \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing .1 \
    --optimizer adam \
    --lr .001 \
    --clip-norm 1 \
    --batch-size 128 \
    --max-update 4000 \
    --no-epoch-checkpoints \
    --save-dir part5a
```
**Development set WER: 18.95
Test set WER: 15.47**

### 5b: Added arguments for 2 d/encoder layers
```
fairseq-train \
    data-bin \
    --source-lang pol.nomsg \
    --target-lang pol.gensg \
    --seed 33 \
    --arch lstm \
    --encoder-bidirectional \
    --dropout .2 \
    --encoder-embed-dim 128 \
    --decoder-embed-dim 128 \
    --decoder-out-embed-dim 128 \
    --encoder-hidden-size 512 \
    --decoder-hidden-size 512 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing .1 \
    --optimizer adam \
    --lr .001 \
    --clip-norm 1 \
    --batch-size 128 \
    --max-update 4000 \
    --no-epoch-checkpoints \
    --save-dir part5b
```
**Development set WER: 14.44
Test set WER: 13.96**

### 5c: Changed the size of the encoder to 256, half of the previous
```
fairseq-train \
    data-bin \
    --source-lang pol.nomsg \
    --target-lang pol.gensg \
    --seed 1555 \
    --arch lstm \
    --encoder-bidirectional \
    --dropout .2 \
    --encoder-embed-dim 128 \
    --decoder-embed-dim 128 \
    --decoder-out-embed-dim 128 \
    --encoder-hidden-size 256 \
    --decoder-hidden-size 256 \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing .1 \
    --optimizer adam \
    --lr .001 \
    --clip-norm 1 \
    --batch-size 128 \
    --max-update 4000 \
    --no-epoch-checkpoints \
    --save-dir part5c
```
**Development set WER: 14.89<br />
Test set WER: 13.51**

----
*Which model had the best development set?*<br />
My model in 5b had the best dev set.<br />
<br />
*Which model had the best test set?*<br />
My model in 5c had the best test set.

## PART 6: Reflections and Stretch Goals

The hint to move obsolete files out of the root was very good.  I had trouble reminding myself which is the source<br />
and which is the target.  It's very easy to switch in pre-processing, so I'm going to double-check all steps for<br />
any errors in calling the correct files.
<br />
Some of the word errors and noted successes are as follows:<br />
<br />
Incorrecly predicting: na > dzi (FEM) (very strange, I'll look into this form) (perhaps because it's a loan word?)<br />
Dąbrowski (2001) mentions that borrowings tend to be uninflected, ending with a -u, or sometimes -a
```  
S-320	k i l o w a t o g o d z i n a
T-320	k i l o w a t o g o d z i n y
H-320	-0.35996171832084656	k i l o w a t o g o d z i d z i
```
    
Incorrectly predicting a -u (MASC;ANIM)
    
```
S-272	h i p o p o t a m
T-272	h i p o p o t a m a
H-272	-0.18618665635585785	h i p o p o t a m u
```

Incorrectly predicting a -u (MASC;HUM)
    
```
S-281	ś w i n i o p a s
T-281	ś w i n i o p a s a
H-281	-0.2113792598247528	ś w i n i o p a s u
```

Incorrectly predicting an -a (MASC;INANIMATE)
    
```
S-196	k o c i o k w i k
T-196	k o c i o k w i k u
H-196	-0.15433460474014282	k o c i o k w i k a
```

Correct!  (MASC;INANIMATE)
```
S-106	w y s o k o ś c i o w i e c
T-106	w y s o k o ś c i o w c a
H-106	-0.11517498642206192	w y s o k o ś c i o w c a
```
    

As we predicted, it would have trouble with irregularity in Masculine forms.<br />
<br />
There were no issues with randomizing, splitting, preprocessing.<br />
The changes I made to prepare.py are as follows:<br />

*This attached the third column (Gender and the animacy of the nouns to both the source and the target)*
```
for nom, gen, anim in source:
    print(" ".join(nom)+" "+ anim, file=n)
    print(" ".join(gen)+" "+ anim, file=g)
```

I used `wc -l` again to make certain the number of rows was correct. <br />   
I manually checked the files so that the form was:<br />
<br />
    
|NOM|GEN|
|-------------------------|----------------------------|
|c z e r n i a k MASC;ANIM|c z e r n i a k a MASC;ANIM|
|k o p t MASC;HUM|k o p t a MASC;HUM|
|s t r z e m i e n n y MASC;ANIM|s t r z e m i e n n e g o MASC;ANIM|
|a n a c h r o n i z m MASC;INAN|a n a c h r o n i z m u MASC;INAN|
|l o s MASC;INAN|l o s u MASC;INAN|
|t y t u ł MASC;INAN|t y t u ł u MASC;INAN|


**Development set WER: 7.82<br />
Test set WER: 9.01**

**Fantastic improvement.**

I ran the first model again, just to be certain:
**
Development set WER: 13.68<br />
Test set WER: 16.22**

### PART 6b : --arch transformer experiment with gender/animacy data attached<br />
this resource helped a bit with parameters: https://huggingface.co/transformers/model_doc/fsmt.html<br />
    also this: https://fairseq.readthedocs.io/en/v0.7.0/models.html

```
fairseq-train \
    data-bin \
    --source-lang pol.nomsg \
    --target-lang pol.gensg \
    --activation-fn relu \
    --attention-dropout 0.15 --activation-dropout 0.15 \
    --seed 410 \
    --arch transformer \
    --dropout .2 \
    --encoder-layers 4 \
    --decoder-layers 4 \
    --encoder-embed-dim 128 \
    --decoder-embed-dim 128 \
    --encoder-attention-heads 4 \
    --decoder-attention-heads 4 \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy \
    --warmup-updates 500 \
    --label-smoothing .1 \
    --optimizer adam \
    --clip-norm 1 \
    --batch-size 512 \
    --max-update 2000 \
    --no-epoch-checkpoints \
    --save-dir part6c
```
    
Part6b - parameter experiment - failed.<br />
First try (this turned out to be very slow.. over 2 hours)<br />
With incomplete parameters:<br />
Development set WER: Errors<br />
Test set WER: Errors<br />
    
Part 6c - An hour, no luck with the above parameters.   <br />
With the parameters above:<br />
Development set WER: Errors<br />
Test set WER: Errors<br />

*Notes:*
Didn't see the warmup updates (500)<br />
I changed max-update to 2000 to halve the processing time.  If successful, I'll change it back.<br />
Slightly worried that even with all of the parameters above, the checkpoint size is the same as the very bad one.<br />
After > 150 epochs, the loss is still near 5.0 - not promising at ~11 updates per epoch.<br />
With the above parameters the Hypothesis does not change from row to row; I'm likely missing something essential.<br />
 







```
