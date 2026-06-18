- ##Frontend framework decision
  
-astrojs vs next js , rn im using astrojs with a serving api for simplicity

-https://machinelearningatscale.substack.com/p/embedding-features-in-weights-to?r=jeeym&triedRedirect=true 
read this about optimized two tower model

- ##sadly have to add outh , wanted to keep it without login :(
  
-imp: if i dont make it login based (no outh ) then i would have to start over again for each person using it
at start i would genrally prefer using content based filtering (cosine similarity)

but that should always be at the start of the session , and with no ouath , i think it'd be hard for me
cuz i would have to start with simple cosine similarity from cold-start.

this wont be favourable , so consider having an outh , i get it that astrojs will need a bit more for this unlike next js but yeah


- ##USER VS ITEM based CBF


<img width="932" height="1412" alt="image" src="https://github.com/user-attachments/assets/c7e8e0b8-e8a6-40fb-9dab-f87e9d613120" />

TLDR :
-i wanna use Item based CBF until a threshold of user data/queiries

-once i get the threshold , i would reset it and then switch to  userbased CBF for a while

-i keep the UCBF until i reach the threshold again(since i reset it earlier)

-and look this threshold is dynamic.....sometimes you need a higher treshold to switch CBF mechanism

and sometimes a lower threshold to switch the CBF option

-IMP: i can decide this by using my UI to make it such a way that the user CAN NOT see the next suggestion until he marks "THUMBS UP or THUMBS DOWN" to current recommendation......

study this threshold thing and plan better lol

(down below my raw thinking as i thought about it)

i thought i can use item based collaborative filtering more initially and until i corss a threshold of user data then i can try user based CBF for a while and reset this threshold , make user explore his own preferences for a bit and once we again hit the threshold we switch to item based collaborative filtering

now , look humans want a mix of whats familiar to them and something novel

so finding the right balance is the key. 

do you think this threshold should be dynamic or static ? , like lets say a user responded well to the user based predictions then do i make the threshold higher or lower

similarly for the item based collaborative filtering , do i increase the threshold or decrease the threshold if the user reacts well to the suggestions

(for suggestions i can use a strict thumbs up and thumbs down before making the user see his next preference)



- ##HALF DIVISION (beta)

keep half of the recommendations userbased and the other half as iterm based

