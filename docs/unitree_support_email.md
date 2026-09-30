Subject: R1 — left shoulder pitch (joint 15) delivers ~1/5 the torque of the right; cannot lift the arm

Hello,

We have an R1 whose **left shoulder pitch motor (joint index 15) produces only
about one fifth of the torque of its counterpart**. Measured repeatedly, j15
peaks at 2.0-2.5 Nm where the right shoulder pitch (joint 22) delivers
10.9-11.9 Nm under identical commands in the same message. The joint reports no
error. It executes small, low-load movements normally and stalls as soon as a
motion requires more torque, so it cannot lift the arm against gravity at all.

This is now our blocking issue and the main purpose of this email. Two smaller
questions follow at the end.

Robot serial number: <FILL IN FROM THE LABEL ON THE ROBOT>

--------------------------------------------------------------------------------
1. The fault, measured two ways
--------------------------------------------------------------------------------

All tests drive `rt/lowcmd` directly with `ai` mode released, and command BOTH
shoulder pitch joints in the SAME message with the SAME gains, so the healthy
right joint acts as a control for the left.

**Test A — position control (kp=80, kd=2), nine identical targets +5° to -35°,
1.2 s settle each:**

    target      left j15              right j22
      +5.0     4.38 deg  0.05 Nm      4.64 deg  0.56 Nm
       0.0     0.06      -0.06       -0.02      0.11
      -5.0    -4.15      -0.31       -4.76     -0.25
     -10.0    -4.17      -0.51       -9.29     -0.85     <-- j15 stops here
     -15.0    -4.18      -0.81      -13.66     -1.71
     -35.0    -4.23      -0.51      -33.08     -2.58

    j15: travelled  0.3-8.6 deg, peak |tau_est| 0.81 Nm, max error 30.8 deg
    j22: travelled 37.7 deg,     peak |tau_est| 2.58 Nm, max error  1.9 deg

j15 tracks the first few targets, stalls, and then sits at a **30.8 degree
error while its reported torque stays flat at 0.5-0.8 Nm**. At kp=80 that error
corresponds to a demand of roughly 43 Nm. On an earlier run the same test
produced 0.3 deg of travel with tau_est flat at 0.04 Nm across all nine targets.

**Test B — direct torque, no position loop (kp=0, kd=0.5, feedforward tau
ramped 0 -> 6 Nm on both shoulders simultaneously):**

    commanded 0.75 Nm -> j22 tau_est +0.76 Nm, moved +0.33 deg
    commanded 1.50 Nm -> j22 tau_est +1.33 Nm, moved +3.02 deg
    commanded 0.75 Nm -> j15 tau_est -0.06 Nm, moved  0.00 deg
    commanded 1.50 Nm -> j15 tau_est -0.06 Nm, moved  0.00 deg

With the position controller removed entirely, **j22's reported torque follows
the commanded value closely, while j15's stays at its at-rest reading regardless
of what is commanded.** Reproduced twice consecutively.

**Test C — a whole posture, both arms, mirrored joint angles.** Commanding both
arms to the same natural "hands raised in front of the chest" pose. The pose is
given as mirrored joint ANGLES, not as a Cartesian target solved separately per
arm, so no inverse kinematics is involved and both arms receive the same numbers
with the sign convention the mirrored axes require:

    joint      | L want   L got  L err  L tau | R want   R got  R err  R tau
    sh_pitch   |  -45.0    -2.0   43.0   1.67 |  -45.0   -41.8    3.2   5.10
    sh_roll    |   20.0    18.4    1.6   2.33 |  -20.0   -18.5    1.5   2.64
    sh_yaw     |   -0.0     0.1    0.1   0.58 |    0.0    -0.1    0.1   0.59
    elbow      |   85.0    84.8    0.2   0.45 |   85.0    85.5    0.5   0.72
    wr_roll    |   -0.0     0.2    0.2   0.56 |    0.0     0.1    0.1   0.67

**Four of the five joints reach the commanded angle within 1.6 degrees on BOTH
arms, drawing near-identical torque. Only shoulder pitch differs: j15 missed by
43 degrees while producing 1.67 Nm; j22 arrived within 3.2 degrees using
5.10 Nm.**

The resulting hand positions:

    right hand ended 2 cm from the commanded position
    left  hand ended 24 cm from it (21.6 cm short forward, 10.5 cm low)

The right arm makes the gesture. On the left, everything below the shoulder
poses correctly and the hand simply stays down at the robot's side.

**Test D — j15's torque output is capped.** Driving
both shoulder pitch joints through the same +/-15 deg move, with every other arm
joint held under position control, repeated three times:

    run 1:  j15 travelled 24.9 deg, peak |tau_est|  2.49 Nm
            j22 travelled 19.8 deg, peak |tau_est| 11.54 Nm
    run 2:  j15 travelled 25.5 deg, peak |tau_est|  2.08 Nm
            j22 travelled 20.8 deg, peak |tau_est| 10.90 Nm
    run 3:  j15 travelled 25.0 deg, peak |tau_est|  2.03 Nm
            j22 travelled 20.1 deg, peak |tau_est| 11.94 Nm

**j15 delivers 2.0-2.5 Nm where j22 delivers 10.9-11.9 Nm under identical
commands — roughly one fifth, reproducibly.**

This explains every other observation. j15 executes any motion needing less than
about 2.5 Nm, which is why small moves near the hanging position succeed. It
stalls the moment more is required — so it cannot lift the arm against gravity,
and in Test A it sits at 30.8 deg of error while producing 0.8 Nm.

One caveat we should state: in Test C the two arms ended at slightly different
angles (left 8.9 deg, right 2.1 deg), so gravity loading was not identical. That
difference works against the failing joint, not for it -- the left arm was at the
HIGHER angle and therefore needed MORE torque, while producing five times less.

--------------------------------------------------------------------------------
2. What we have ruled out
--------------------------------------------------------------------------------

  - **Not mechanical.** The arm is free when moved by hand, and a mechanical
    obstruction would show torque rising against the stop. We measure the
    opposite: large position error with near-zero torque.
  - **Not the other joints, and not the arm's power or wiring.** All nine
    remaining arm joints move and draw current normally under the same test.
    In one run the four other LEFT joints (16-19) tracked commanded positions
    to within 3.2 deg **in the same `rt/lowcmd` message in which j15 did not
    move at all**. Whatever is wrong is downstream of the message, the bus and
    the arm's power feed, all of which the working joints share.
  - **Not our control software.** Test B bypasses our inverse kinematics,
    clutch and teleoperation entirely and writes `rt/lowcmd` directly.
  - **Nothing is reported as faulty.** `motorstate` = 0, `mode` = 1,
    temperature 31-41 °C, bus voltage 35.5 V, no error flags on j15.

--------------------------------------------------------------------------------
3. The intermittency, which we think is the important detail
--------------------------------------------------------------------------------

The fault is **not constant, and appears to be worsening**:

  - 2026-08-24: completely dead — 0.04 Nm across a full nine-target sweep.
  - 2026-08-27, after a power cycle: worked normally, reaching 8.0 Nm during
    teleoperation and tracking commands correctly for roughly an hour.
  - 2026-08-27, later the same session: managed 19.9 deg of travel at 1.01 Nm
    on small (+/-12 deg) moves, but stalled on larger ones.
  - 2026-08-27, later still: zero current under direct torque command.

What first looked like random intermittency is, we now believe, the torque cap
in Test C: **the same joint passes or fails depending on how much torque the
motion demands, not on when it is attempted.** Two tests run two minutes apart
gave opposite results and then both reproduced — the low-load sweep succeeded
three times, the lift failed every time. That is consistent behaviour against a
ceiling, not an intermittent connection.

The cap itself has varied between sessions (0.04 Nm on 24 August, 2.0-2.5 Nm
now, and one teleoperation reading of 8.0 Nm), so the ceiling appears to move.
This looks to us like a motor driver or a connection rather than a failed
winding, but we would rather have your diagnosis than act on ours.

Please advise how you would like to proceed — whether this is field-repairable
with a part you can send, or whether the arm or robot needs to be returned.

--------------------------------------------------------------------------------
4. Two smaller questions
--------------------------------------------------------------------------------

**4a. ddq is transmitted as all zeros.** You listed ddq (estimated angular
acceleration) among the fields usable for estimating the load from a lifted
object. In a live capture of `rt/lowstate` just now:

    tau_est     : 24/35 non-zero      OK
    q           : 26/35 non-zero      OK
    dq          : 20/35 non-zero      OK
    vol         : 26/35 non-zero      OK
    temperature : 26/35 non-zero      OK
    ddq         :  0/35 non-zero      NOT POPULATED

Is ddq expected to be populated on our firmware, and will an update populate
it? If not we will differentiate dq ourselves, but we would prefer your
estimate if it is available.

**4b. No stereo / depth camera streams on the DDS bus.** We would like to run
VSLAM. Reading the robot's full topic list, the only vision-related stream is
`rt/frontvideostream` (`Go2FrontVideoData_`), plus the RTSP stream on port
8551. There are no stereo, depth, image or point-cloud topics. There is a
`rt/api/slam_operate/request` service with no data topics feeding it.

Does this unit have binocular cameras, and if so are their streams published
only on the internal ROS 2 network on the Jetson rather than the external DDS
bus? Is running VSLAM on the Jetson the intended approach?

--------------------------------------------------------------------------------
5. Previously reported issues that are now resolved
--------------------------------------------------------------------------------

So that you can disregard them: in our earlier message we reported that
192.168.123.164 did not respond and that the onboard Jetson was unreachable.
**Both are now working.** 192.168.123.164 answers and TCP port 22 is open;
`ssh unitree@192.168.123.164` reaches the onboard computer. No action needed.

Our DDS access is otherwise healthy: 18 applications and 91 topics visible,
including `rt/lowstate`, `rt/lowcmd`, `rt/arm_sdk`, `rt/handcmd` and
`rt/dex3/*/state`, with live joint telemetry, IMU and battery voltage all
reading correctly over the wired connection.

Thank you for your help.

Best regards,
Kanak Pandit
ARENA2036, Stuttgart
