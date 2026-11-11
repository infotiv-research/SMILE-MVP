    Operational scenarios 

This analysis focuses on two representative scenarios, each with specific functional and safety goals. 

    Scenario 1: Robot Operation at Intersections 

    Preconditions 

    The robot operates within a defined intersection area connecting multiple drivable paths 

    Forklifts may approach the intersection from multiple directions 

    The system includes perception modules capable of detecting and tracking forklifts, including trajectory and manoeuvre classification 

    A prediction horizon (e.g., 1–3 seconds) is available for estimating forklift motion and time to intersection 

    Potential Hazards 

    H1: Late or missed detection of a forklift approaching the intersection 

    H2: Incorrect trajectory classification (conflicting vs. non-conflicting path) 

    H3: Incorrect manoeuvre classification (moving, loading, stationary) 

    H4: Robot remains in the intersection while a forklift is about to enter 

    H5: Free-space detection failure (if used as primary input) 

    Mitigation Actions 

The robot shall implement the following behaviours: 

    Entry control: Do not enter the intersection if a forklift is detected within a defined approach distance (e.g., 7–8 m) and is predicted to enter within a short time horizon 

    Intersection clearance: Immediately vacate the intersection if a forklift approaches within the same threshold and shows intent to enter 

    Occupancy safety rule: Do not remain in the intersection if a forklift could reach it within ~1 second 

    Trajectory-based response:  

    Conflicting path: yield / delay entry 

    Non-conflicting path: proceed with caution 

    Manoeuvre-based response 

    Loading within intersection: treat as obstruction, avoid entry 

    Stationary forklift: proceed only if path is not blocked 

    Fail-safe behaviour: If uncertainty or anomalies are greater than threshold (e.g., perception uncertainty, free-space uncertainty), stop or remain outside the intersection 

    Scenario 2: Robot Operation Outside Intersections 

    Preconditions 

    The robot operates along predefined routes free from known stationary obstacles 

    Alternative paths are available (by planning module) to bypass temporary obstacles 

    Dynamic obstacles (forklifts, humans, movable objects) may appear along the route 

    The system can estimate time-to-collision (TTC) for moving objects 

    Potential Hazards 

    H1: Collision with a moving forklift (e.g., head-on or crossing path) 

    H2: Collision due to incorrect TTC estimation or trajectory prediction 

    H3: Collision with humans or temporarily stationary movable objects 

    H4: Misclassification of forklift manoeuvre (e.g., loading vs. stationary)  

    Mitigation Actions 

The robot shall implement the following behaviours: 

    Collision risk monitoring: Continuously evaluate overlap between the robot’s planned path and predicted forklift motion within a defined TTC window (e.g., 3 seconds) 

    Dynamic obstacle avoidance: Avoid collisions with humans and movable objects, including temporarily stationary ones 

    Head-on interaction: Move aside and transition to a safe parked state when a forklift approaches head-on 

    Loading forklift handling 

    Notify the forklift operator 

    Proceed with caution if safe to do so 

    Stationary forklift handling: Continue along the planned path if no collision risk is detected 

    Fail-safe behaviour: If perception uncertainty or anomalies exceed acceptable thresholds, slow down, stop, or park safely 

 

  

    Scenario 3: Temporary or unknown obstacle 

    Preconditions 

    AMR operates within the validated ODD 

    Temporary obstacles may appear in the drivable area 

    Alternative route or stopping capability is available 

    Potential Hazards 

    Undetected obstacle in drivable space 

    Incorrect free-space classification 

    Mitigation Actions 

    Treat unknown objects as obstacles 

    Replan robot route 

    Trigger safe mode if trustworthiness becomes lower than threshold 

    Safety goals  

ID 
	

Description 

SO-01 
	

Prevent collision with forklifts 

SO-02 
	

Prevent unsafe occupancy of intersections 

SO-03 
	

Prevent collision with unknown or temporary obstacles 

SO-04 
	

Detect operation outside validated ODD 

SO-05 
	

Transition to safe mode when trustworthiness is insufficient 