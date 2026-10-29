"""Navigation: drive to a spot on the lidar map, or rove and explore it.

The pieces, each a module, each testable on its own with no car:

  mapview    the map and pose the planner is handed (MapView), and the
             truck-centred LocalMap it falls back to when the global map is
             not trusted. The seam to the live SLAM is here.
  truck      the truck's measured size, its turning circle, and the one
             conversion from a path curvature to a steering command.
  costmap    obstacles grown by the truck's half-width plus a margin, a cost
             gradient beyond that, and the distance field the planner and the
             footprint check read.
  planner    hybrid A*: arcs the truck can actually drive, forward and
             reverse, so a turn too tight for one sweep becomes a
             three-point turn.
  follower   pure pursuit for the counter-phase four-wheel steer, fed the
             pose every decision, with a steering gain it learns as it goes.
  frontier   where the known map meets the unknown, for rove.
  navigator  the goal handling: a clicked spot or rove, re-planning when the
             path is blocked, giving up, and asking for the back-out when it
             is stuck.
  click      the Foxglove side: a goal clicked in the 3D panel.
  driver     NavDriver, the autopilot chassis with the navigator deciding.
             Every safety layer the rules and the policy sit on -- the
             forward floor, the rear stop, STUCK and the back-out -- stays
             underneath it.

Frames: the map frame, metres, x and y, yaw counter-clockwise, as the live
SLAM (crawler/mapping) writes it. The pose is the lidar's, which sits about
the middle of the truck between the bumpers. Grids are indexed [ix, iy], x
first, with cell (0, 0)'s outer corner at `origin`.

Nothing here runs on the car unless auto is started with --nav, and --nav is
off by default.
"""
