# How to write data driven blog post? - Dota 2 Analysis

This project was created as a part of [Data Scientist Nanodegree](https://eu.udacity.com/course/data-scientist-nanodegree--nd025)
at [Udacity](https://eu.udacity.com/?cjevent=d1a59cbeab1111e9834e02630a18050b), for the first project “Write A Data
Science Blog Post”. Goal of this project is to present my ability of getting access to data, analyzing it, understanding 
and presenting in elegant way.

This project is compatible with `CRISPDM (Cross-Industry Standard Process For Data Mining)` methodology. 
Consequently it's structure will consists of the following parts:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Data Modeling
5. Results presentation

## Getting Started

### Files Overview
Thinking process preview:
- [CRISP-DM pipeline.ipynb](notebook/CRISP-DM%20Pipeline.ipynb) - showcase notebook where the way how data was fetched and
how the thinking process during answering business questions looked is presented

Backend side:
- [generate_report.py](generate_report.py) - script that fetches, transforms and cleans the data, saves it to [
data](data) dir, it generates graphs which are an answer to stated business questions and saves them in [image](image)
dir

Fronted side:
- blog post available under following [LINK](https://medium.com/@krzyk.kamil/dota-2-valid-career-path-or-just-extraordinary-form-of-entertainment-91c456ea82fc)

Other:
- As an exception, downloaded data (in folder [data](data)) and generated graphs (in folder [image](image)) used in
blog post were kept in this repository. This is because data changes in realtime and same results might not be reproduced in the future.


### Prerequisites

Project requires Python 3.6.6 with installed [virtualenv](https://pypi.org/project/virtualenv/).

### Setup

Dependencies setup:
1. Pull the project to `<project_dir>`.
2. Setup new virtualenv with Python 3.6.6.
3. Install requirements.txt file via pip.
4. Setup project package by navigating to `<project_dir>` and running:
```
$ python setup.py develop
```


Result reproduction:
1. Navigate to `<project_dir>` and run:
```
$ python generate_report.py
```


Thinking process preview setup:
1. Run jupyter notebook instance from terminal by invoking `jupyter notebook` command. (available by default 
2. http://localhost:8888 in any browser)
3. Navigate to `<project_dir>/notebook`.
4. Launch `CRISP-DM pipeline.ipynb`

## Stated Problem
In this project I have decided to take a closer look at [Dota 2](http://blog.dota2.com/) professional scene. It is most 
played game on [Steam](https://steamcommunity.com/) and takes 
[7th place in the ranking of most played games in the world](https://newzoo.com/insights/rankings/top-20-core-pc-games/?source=post_page---------------------------).

As a former pro-aiming player I am interested how the scene has changed in last 7 years since the time when I was 
active. I would like to find answers to the following questions:

---
#### - How is the game doing? Is the game losing popularity?
![](image/activity_plot.png?raw=true)

It is possible to observe that average amount of players online has been raising over first two and half years of game 
release. Since that moment it holds on the same level of around 500 thousands of players online players per hour. The 
game vitality seems preety stable and lately started sligtly rising what can be explained by upcoming world championship 
TI 2019 in August 2019. Furthermore the game has introduced Battle Pass in May, which allows players to grind additional
in-game bonuses by completing prepared subtasks and quests.

---
#### - How many players are there? What part of them plays ranked matches? How many of them are professional players?
![](image/player_count.png?raw=true)

Professional players are 0.00009% of all players.

---
#### - What is average MMR (Match Making Ranking) of common Dota 2 player?
![](image/mmr.png?raw=true)

Graph shows the MMR distribution in player group which decided to play Ranked Match mode. Interesting observation in 
this graph are peaks at mmr values such as 3000, 4000, 5000 and also small peaks at 2000 and 6000. This is most likely
caused by the fact that people tend to climb the ladder and they set those "complete" numbers as their final goals. It 
is very unlikely and hard to improve the personal ranking over 2000 points in one season. Average player MMR is: 2929.

---
#### - What is average division of common Dota 2 player?
![](image/ranks.png?raw=true)

The same MMR distribution can be translated to 8 ingame divisions, where each divisions has 7 separate level. Highest 
division called "Immortal" is for players that have achieved very high score and top 1000 places in ranking is displayed 
on the division herb.

In general each player has two kind of MMR scores - solo and team. Player division is decided based on highest MMR
value.

The visible accumulation of players at Ancient VII is caused by the fact, that players cannot reach Divine division 
through team MMR. It has to be done alone, only through playing Solo Rank Matches.

---
#### - How does professional player performance compare to average player?
![](image/pro_player_mmr.png?raw=true)

Average MMR of professional player is 6027. It is almost two times higher than average MMR of average Dota 2 player.
Definition of when someone can be called professional player is unclear. There are players that earn money by teaching
others to play Dota via couching sessions or streaming. Those people could be called professionals because they earn
money by playing the game. It seems that for OpenDota, player is tagged as professional if he played in tournament 
that offers larger money prize. Still within a group of all professional players listed by OpenDota, not everyone 
are currently a part of complete 5-man team. Those who are currently active seems to have higher average MMR of 6777.

---
#### - How much professional players can earn from winning tournaments?
![](image/ti_prizepool.png?raw=true)

## Built With

* [Udacity](https://www.udacity.com/) - project for passing Data Engineering section and Data Science Nanodegree
* [OpenDota](https://www.opendota.com/) - data contributors

## License

This project is licensed under the MIT License.