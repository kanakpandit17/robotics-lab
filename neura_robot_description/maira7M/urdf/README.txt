# Maira

- maira_macro.xacro : macro that contains structure of robot
- maira_limits : contains information about joint limits, joint velocity limits and joint effort maira_limits
- maira_kinematic_parameters_nomial : contains the nominal kinematic parameters 
- maira_dynamic_parameters_nominal : contains the nominal dynamic parameters

exchange these files after calibration and recreate description files (urdf, sdf, ...)

to create the urdf :

xacro maira.urdf.xacro > maira.urdf

