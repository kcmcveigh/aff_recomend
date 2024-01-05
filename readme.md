# Affective Recommendations

As a psychologist I've been interested in recommendation systems for a while now for several reasons. The first being that at a societal level they're ubiquitous and are often the "algorithm" which people refer to as colloquially controlling our lives. Next being that I think as a tool for the psychological science they have tremendous potential. I think this because unlike much of psychology which searches for effects that hold for groups of people recommendation systems seem to hold a lot of potential at examining how combination of individual factors, as well as what I think of as situational factors, which in I think fit nicely into the items portion of many recommendation factors, combine to create behavior, and psychological experience. This repo is my first foray into testing out some recommendation systems.

## Collaborative Filtering

I start very simply piggybacking on FastAI implemenation of a classic matrix factorization implementation. Under the hood this implementation creates a simple pytorch class which decomposes one's participants (or in recommendation system terms "users") x situations (recsys terms "items") matrix, into a set of set of latent factors for each participant a essentially a P x N  (latent factor) matrix we'll all P a S (situation) by N matrix well call S as well as a bias term for each participant (a P x 1 matrix, Pb) and situation (S X 1 matrix Sb). The predicted rating (or whatever target variable you've used for your matrix is predicted by) 

predicted_rating = dot(P<sub>current_participant</sub>,S<sub>current_situatio</sub>) + Pb<sub>current_participant</sub> + Sb<sub>current_situatio</sub>

I train this model on a dataset from our lab which has ~100 people watching about ~30 short videos while we measure their physiology and have them make fear ratings after each video. I think the Collaborative Filtering Recommendation System set up already is quite naturally suited to this data. For instance one issue we often deal with in self report data is that people will use the scales differently. For example one person may rate their fear experiences using just a small part of the scale (it's hard to evoke fear experiences in the lab that are the same as a near death experience) while another participant may use the full scale. It may be that the relative pattern of their fear ratings across videos are quite similar but their absolute ratings are quite different. This way of decomposing the rating matrix can already account to this for some extent by seperating some of the asbolute magnitude into the bias terms (intercepts) while the participant factors may capture relative similarities (particularly depending on what matric you use to measure that similarity). Of course there are many other techniques that can do this too however relatively few are used in the psychology take into account item effects (i.e. responses may be very item dependent not just person dependent, although item response theory springs to mind as a classic approach to this).


## The code itself

The main action for training the models is the fastai_cf.py
