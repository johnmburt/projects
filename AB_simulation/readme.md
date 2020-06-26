# A/B testing user preference simulation 

## John Burt

### Project goal:

This project came out of discussions over how our Data Science meetup group could host an A/B testing workshop. The problem is that A/B testing is an iterative experimentation and data collection process: you present users with a website configured in two different ways, record their response (Click Throughs, Conversions, etc), compare the responses, make improvements and then conduct another experiment, etc. So, if you want people to learn and explore A/B testing, you can't just provide a static dataset - you need to provide a simulation environment that testers can probe with their own A/B tests. 

My initial solution was to create a simple model that simulates preferences for website design elements among a population of simulated users (sim-users). The model could be hosted as a web app or microservice.  An administrator would build a sim-website and specify sim-user preferences to the elements of the site (design elements, CTAs, etc). These preference would be unknown to workshop participants and their job would be to run experiments to find them out by running experimental trials, receiving user responses and then running new trials.

### Step 1: EDA using real preference data

[EDA notebook](consumer_preference_EDA.ipynb)

The first thing I realized was that for the model to be realistic, sim-user preferences should be variable, but predictable. So I started thinking in terms of a population of users having an underlying distribution of preferences for each design element on a website. To keep things simple, I would like to assume this is a normal distribution. But is that a valid assumption? To test that I located some food preference data posted on the [Sensometric Society Data Set Repository](http://www.sensometric.org/datasets) and I plotted out the distributions of preferences for two experiments (brown bread, and fried mozarella cheese sticks). What I found there was that yes, preferences tend toward a normal distribution, though it is often slightly bimodal. The bimodality of preferences was interesting and according to one co-author of these studies was due to a polarity effect (think Coke vs Pepsi people). That's very interesting and worth pursuing later, but for my initial model I will go with the main trend of a normal distribution of perferences for each simulated website element.

### Step 2 the initial model

[Model notebook](consumer_pref_sim_1_vf.ipynb)

For the first model, I wanted to keep things as simple as possible, while still allowing a reasonably realistic output. To do that I created a simulation of user preferences with these features:

- **The simulation emulates a population of users who have pre-defined preferences for the website elements that will be tested.** For example, given red, green and blue as background colors, we can specify that users tend to dislike red and green, and prefer blue backgrounds. 

- **Simulated user preferences are modeled on a normal distribution.** Preference for each element level (e.g., "blue background") is specified by two terms: mean and variance. When a sim-user is tested in a trial, their preferences are randomly selected based on the distribution described by those terms. This results in a more stochastic and realistic distribution of responses.

- **A sim-user's overall website preference score is based on the average of its preferences for all of the website element levels it's presented with.**

- **An importance factor is applied when generating responses.** Only a fraction of a real user's decision to click on a buy button will be due to the website design. I call this factor importance: if the website design is very inflnential, then importance will be higher.

- **Response is calculated using a simple threshold.** The dependent variable for the model is conversion. If the final score is higher than the defined response threshold, then the result will be yes (conversion), otherwise no.

