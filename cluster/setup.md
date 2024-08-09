env = est.Moments_estimation(length = 1000, 
                              om0 = om0, 
                              noise = noise,
                              cs = cs,
                              max_time = 30,
                              penalty = -20,
                              time_step = 0.25,
                              min_time = 1)