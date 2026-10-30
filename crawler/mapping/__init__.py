"""Live 2D lidar SLAM: the room's map, built from nothing as the truck drives.

Off unless the truck is started with --lidar --slam, and mapping only:
nothing here commands the truck.

The parts
---------
  scan.py     a revolution's 720 ranges -> car-frame points with line
              normals, de-skew, and the maybe-ground mask for a tilted plane
  imu.py      the OAK-D IMU as heading rate about gravity, and tilt
  clock.py    when a revolution really ended, from the sensor's own stamps
  matcher.py  point-to-line ICP, and how well the fit pins the pose down
  grid.py     the map: int8 log-odds, 5 cm cells, a 102 m window that
              grows to 205 m and only then scrolls
  slam.py     Slam.step(), one scan: predict, de-skew, match, gate, write
  runner.py   SlamRunner: Slam on a worker beside the control loop, and
              the Foxglove messages (telemetry/map_view.py draws them)
  mapfile.py  the map on disk (log/slam_map.npz): saved as it goes and on
              a clean stop, restored on boot for the truck to find itself on

tools/slam_replay.py runs the same Slam over a recording; tools/slam_bench.py
times it on the Pi.

The interface a planner builds on
---------------------------------
Everything comes off the SlamRunner main() made (it is the `slam` handed to
lidar.LidarReader), and every call returns at once:

  runner.latest()   the newest slam.Result: the pose (x, y metres, yaw
                    radians, in the map frame: origin and heading where the
                    truck booted) and what to trust it by. state ==
                    "tracking" and matched is a pose worth steering by;
                    anything else (tilted, rolled, lost, no_points) or a low
                    `constraint` (a corridor, an open space) is not, and the
                    fallback is the lidar's own car-frame scan
                    (LidarReader.latest().bins) and following a person.
                    The policy's map input also drives on tilted and
                    matched: the map as last written, at a pose still
                    tracked (egomap.trusted).
  runner.pose()     (x, y, yaw) of that, or None
  runner.map()      a MapView, at most a second old: uint8 codes per 10 cm
                    pixel, grid.UNKNOWN / FREE / OCCUPIED, and where pixel
                    (0, 0) sits in the map frame. Unknown is not free.
  runner.path()     the (x, y) points driven

For a cost map: inflate OCCUPIED by the truck's half width,
lidar.HALF_WIDTH_MM (136 mm) plus a margin, treat UNKNOWN as blocked or
expensive, and search (A*) on the 10 cm pixels; a 4WS follower then tracks
the path against runner.pose(). A goal clicked in Foxglove's 3D panel in the
"map" frame is already in these coordinates. A process worker sends the map
as these pictures because they are small; a planner that wants the 5 cm
log-odds runs with the thread worker and reads Slam.grid
(Grid.log_odds(), Grid.occupied()) on that thread's object.

What it does not do
-------------------
No loop closure: the error of a long drive is never taken back out, only
kept small. Slam._write(pose, points) is the one place a scan enters the
map, which is where a pose graph would log scans to re-render from.
Relocalisation only near where the truck froze (reloc.SEARCH_R_M, or the
shelf's reckoned doubt): a truck carried further than that while lost is
not found until it drives back near.
The map keeps everything inside 205 m a side; past that it scrolls and
forgets, and says so in the log as an error.
"""
