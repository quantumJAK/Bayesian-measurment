import numpy as np
import matplotlib.pyplot as plt

min_std = 0.01

def get_std(probs, grid):
    if 0 > np.sum(probs*grid**2)-np.sum(probs*grid)**2:
        return min_std 
    else:
        return np.sqrt(np.sum(probs*grid**2)-np.sum(probs*grid)**2)
    

            
def get_estimate(probs, grid):
    return np.sum(probs*grid)
    ind = np.argmax(probs)
    return grid[ind]


class game():
    '''
    Function that play the game for a number of episodes and return the rewards, actions, mus, stds and oms
    Input:
        episodes: number of episodes to play
        model: model to use to play the game
        env: environment to play the game
    Output:
        rewards: rewards obtained for each episode
        actions: actions taken for each episode
        mus: mus of the field for each episode
        stds: stds of the field for each episode
        oms: oms of the field for each episode
    '''
    def __init__(self, episodes, env, model=None, policy = None, **kwargs):
        env.reset()
        env.rng_shot = np.random.default_rng(env.seed_shot)
        env.rng_field = np.random.default_rng(env.seed_field)
        self.freq_grid = env.freq_grid
        estimation_length = env.estimation_length
        self.rewards = np.zeros((episodes, estimation_length))
        self.stds = np.zeros((episodes, estimation_length))
        self.actions = np.zeros((2,episodes, estimation_length))
        self.mus = np.zeros((episodes, estimation_length))
        self.oms = np.zeros((episodes, estimation_length))
        self.play(episodes, env, policy, model, **kwargs)
    
    def play(self, episodes, env, policy = None, model = None, **kwargs):
        single_action = False
        for episode in range(0, episodes):
            done = False
            k = 0
            n_state = env.reset()[0]
            while not done:
                if model:
                    action,_ = model.predict(n_state)
                    
                elif policy:
                    action = policy(n_state, **kwargs)
                    single_action = True
                else:
                    action = env.action_space.sample()

                n_state, reward, done, _, info = env.step(action)
                weights = n_state
                print(self.freq_grid)
                plt.plot(weights)
                self.mus[episode,k] = get_estimate(weights, self.freq_grid)
                self.stds[episode,k] = get_std(weights, self.freq_grid)
                self.rewards[episode,k] = reward
            
                #check if action is a array or a in
                try: 
                    self.actions[0, episode,k] = action[0]
                    self.actions[1, episode,k] = action[1]
                except IndexError:
                    self.actions[episode,k] = action
                self.oms[episode,k] = info['om']
                k = k+1

    def get_error(self):
        return np.abs(np.abs(self.oms) - np.abs(self.mus))




class CMA_optimizer(): 
    '''
    This is the optimization algorithm that will parametrize the Vin array using two parameters (Vmax, std).
    This would significantly limit space of parameters      
    '''

    def __init__(self, generations, population_size, bounds, sigma, mean, policy, episodes, env, opt_seed = 1,
                seed_estimation = 2):
        '''
        Constructor of the cma class. It initializes the CMA-ES algorithm and sets its parameters.
        ----------
        Arguments:
            ComsolModel {ComsolModel} -- ComsolModel class object.
            generations {int} -- Number of generations to run the algorithm.
            population_size {int} -- Number of probes to evaluate per generation.
            bounds {np.array} -- Bounds of the parameters to optimize.
            sigma {float} -- Initial standard deviation of the parameters.
            mean {np.array} -- Initial guess of the parameters.
        '''
        from cmaes import CMA
        self.optimizer = CMA(mean=mean, bounds = bounds, sigma=sigma , population_size=population_size, seed=opt_seed)
        self.generations = generations
        self.policy = policy
        self.episodes = episodes
        self.env = env
        self.rng_est = np.random.default_rng(seed_estimation)


    def optimize(self):
        ''' 
        Main optimization loop. It runs the arguments and the cost function for all evaluated points

        ----------
        Arguments:
            None
        Returns:
            solutions_all {list} -- List of all evaluated points and their cost function.
        '''
        solutions = []
        for gen in range(self.generations):
            print(gen)
            population = self.ask_for_batch(self.optimizer.population_size)
            f_probes = self.evaluate_batch(population)
            solutions.append(f_probes)
        self.solutions = np.array(solutions)
        return self.solutions
    
    def get_representative(self):
        '''
        Function to extract the best solution from the optimization algorithm.
        ----------
        Arguments:
            None
        Returns:
            best_solution {np.array} -- Best solution found by the algorithm.
        '''
        result = np.mean(self.solutions[-1], axis=0)
        f_avg = result[-1]
        x_avg = result[:-1]
        return x_avg, f_avg
        

    def ask_for_batch(self, N):
        '''
        Function to draw the population of N samples from the current distribution
        ----------
        Arguments:
            N {int} -- Number of probes to evaluate.
        Returns:
            population {list} -- List of probes to evaluate.
        '''
        population = []
        for n in range(N):
            population.append(self.optimizer.ask())
        return population
    

    def evaluate_batch(self, population):
        '''
        Function to evaluate the cost function for the population of probes. First it extracts the features (y) and translate them into fitness (cost function).
        ----------
        Arguments:
            population {list} -- List of probes to evaluate.
        Returns:
            solutions {list} -- List of evaluated probes and their cost function.
        '''

        solutions = []
        solutions_to_return = []
        for p in population:
            fit = self.cost_function(p)
            solutions.append(((p,fit)))
            solutions_to_return.append(np.hstack((p,fit)))
        self.optimizer.tell(solutions)
        return np.array(solutions_to_return)
    

    def cost_function(self, y):
        '''
        WE WANT TO IMPLEMENT THIS FUNCTION 
        
        y = f(x) where x is the parameters and y is the output of comsol (features).

        Cost_function = g(y) cost function is a function of the features
        '''
        return -np.sum(game(episodes = self.episodes, env = self.env , 
        policy = self.policy, rng_est = self.rng_est, x=y).rewards)/self.episodes


from scipy import optimize
def policy_flip(state, **kwargs):
    return 0

def policy_random_p(state, *args, **kwargs):
    rng = kwargs["rng_est"]
    x = kwargs["x"]
    norm = sum(x)
    pflip = x[0]/norm
    pest =  x[1]/norm
    pcheck = x[2]/norm

    return rng.choice([0,1,2], p=[pflip,pest,pcheck])


def policy_random_p2(state, *args, **kwargs):
    rng = kwargs["rng_est"]
    x = kwargs["x"]
    norm = sum(x)
    pflip = x[0]/norm
    pest =  x[1]/norm

    return rng.choice([0,1], p=[pflip,pest])


def policy_random(state, *args, **kwargs):
    rng = kwargs["rng_est"]
    return rng.choice([0,1,2], p=[1/3,1/3,1/3])

def policy_max_std(state, *args, **kwargs):
    rng = kwargs["rng_est"]
    x = kwargs["x"]
    std_max = x[0]
    pest = x[1]
    std = state[1]
    mu = state[0]
    if state[0] > 0:
        if std/state[0] < std_max:
            return 0
    
    return rng.choice([1,2], p=[pest,1-pest])

def policy_min_mu(state, *args, **kwargs):
    rng = kwargs["rng_est"]
    x = kwargs["x"]
    mu_min = x[0]
    pest = x[1]
    mu = state[0]

    if mu > mu_min:
        return 0
    else:
        return rng.choice([1,2], p=[pest,1-pest])

def policy_interval(state, *args, **kwargs):
    rng = kwargs["rng_est"]
    x = kwargs["x"]
    n = state[2]
    Nflip = np.floor(x[0]) 
    Nest = np.floor(x[1]) 
    Ncheck = np.floor(x[2])

    if n % (Nflip+Nest+Ncheck) < Nflip:
        return 0
    elif n % (Nflip+Nest+Ncheck) < Nflip+Nest:
        return 1
    else:
        return 2




