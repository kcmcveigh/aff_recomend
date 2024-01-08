# Affective Recommendations



## In brief + Plus Results

In this repo I train a collaborative filtering model on a dataset from our lab which has ~100 people watching about ~30 short videos while we measure their physiology and have them make fear ratings after each video. The most classic application of this would be to to predict the fear rating (explicit feedback) from both the participant id and movie id. I train and validate a model to do this, but also train two additional models which predict different sorts of physiological parameters derived while participants watched the videos (Heart Rate, and Skin Conductance Responses). After training all models I compare their recommendations and representations to show that videos that evoke the greatest responses for each type of videos for each person can vary drastically across people (See figure below). These results suggest the power of recommendations systems as we can look model this variance at the individual level, but also that the common assumption made by many lay people, psychologists, and corporations that physiological responses = affective responses isn't true for many of the individuals in the dataset. Below I expand on pretty much every part of the project and my thoughts on it for those interested.

## In some more detail

### A little bit of motivation

As a psychologist I've been interested in recommendation systems for a while now for several reasons. The first being that at a societal level they're ubiquitous and are often the "algorithm" which people refer to as colloquially controlling our lives. Next being that I think as a tool for the psychological science they have tremendous potential. In my view this potential comes from how many recommendations systems model how individual factors, and situational factors (or item factors in recommendation system languages) combine to create behavior, and psychological experience. This repo is my first foray into testing out some recommendation systems.

To explore the use of some these techniques I chose this dataset because I've become increasingly interested in how recommednation systems might be used in the context of precision psychiatry as well as just in time interventions. While the current repo is more or less a toy example and pales in complexity to how these would need to be implemented in practice I think it has some nice features. First it is an affective context which a large body of research shows are highly personalized across individuals such that one situation (in this context a video) may evoke very different affective responses. This highly personalized nature seemed ripe for a recommendation system application. The second is that this dataset contains both physiology and affective data. I've increasingly seen wearable companies motion at the idea of taking some physiological reading inferring someone's affective state (which in my view "stress" falls under), then pushing that person realtime intervention. I was interested in this dataset if were to think of the videos as both situations but also how treaments, how might the recommended videos differ if we focused on the physiology recorded to each video versuses the explicit affective ratings. I think this comparison gets at the question of how well do physiology and affect correlate, or as is fashionable in the affective science community, how much coherence is there between these two different measures. Of course there are many more sophisticaed ways to do both of these things which may work far better then this classic recommendation system technique but I think this still gives us some insight. 

### Intro to Collaborative Filtering

I start very simply piggybacking on FastAI implemenation of a classic matrix factorization implementation. Under the hood this implementation creates a simple pytorch class which decomposes one's participants (or in recommendation system terms "users") x situations (recsys terms "items") matrix, into a set of set of latent factors for each participant a essentially a P x N  (latent factor) matrix we'll all P a S (situation) by N matrix well call S as well as a bias term for each participant (a P x 1 matrix, Pb) and situation (S X 1 matrix Sb). The predicted rating (or whatever target variable you've used for your matrix is predicted by) 

predicted_rating = dot(P<sub>current_participant</sub>,S<sub>current_situatio</sub>) + Pb<sub>current_participant</sub> + Sb<sub>current_situatio</sub>

I train this model on a dataset from our lab which has ~100 people watching about ~30 short videos while we measure their physiology and have them make fear ratings after each video. I think the Collaborative Filtering Recommendation System set up already is quite naturally suited to this data. For instance one issue we often deal with in self report data is that people will use the scales differently. For example one person may rate their fear experiences using just a small part of the scale (it's hard to evoke fear experiences in the lab that are the same as a near death experience) while another participant may use the full scale. It may be that the relative pattern of their fear ratings across videos are quite similar but their absolute ratings are quite different. This way of decomposing the rating matrix can already account to this for some extent by seperating some of the asbolute magnitude into the bias terms (intercepts) while the participant factors may capture relative similarities (particularly depending on what matric you use to measure that similarity). Of course there are many other techniques that can do this too however relatively few are used in the psychology take into account item effects (i.e. responses may be very item dependent not just person dependent, although item response theory springs to mind as a classic approach to this).


### The code itself

The main action for training the models is the fastai_cf.py.


