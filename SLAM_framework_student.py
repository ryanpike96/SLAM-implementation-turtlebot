#!/usr/bin/env python2

import rospy
import numpy as np
import math
import time as t
from collections import defaultdict
import message_filters

#from scipy.linalg import block_diag
# Message types
from nav_msgs.msg import Odometry
from cylinder.msg import cylDataArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import Imu
# Functions
from Relative2AbsolutePose import Relative2AbsolutePose
from Relative2AbsoluteXY import Relative2AbsoluteXY
from Absolute2RelativeXY import Absolute2RelativeXY
from pi2pi import pi2pi
from mapping import mapping
from Error_Function import ErrorFunction
from tf.transformations import quaternion_from_euler
import sys, select, termios, tty
import csv



# landmarks' most recent absolute coordinate
landmark_abs_ = defaultdict(list)
seenLandmarks_ = []
# State Transition Model
F_ = []
# Control-Input Model
W_ = []
# dimension of robot pose
dimR_ = 3


class Robot():

    def __init__(self, pose, pos_Cov, sense_Type):

        self.x = pose[0][0]
        self.y = pose[1][0]
        self.theta = pose[2][0]
        self.poseCovariance = pos_Cov
        self.senseType = sense_Type

    def setPose(self, new_pose):

        self.x = new_pose[0][0]
        self.y = new_pose[1][0]
        self.theta = new_pose[2][0]

    def getPose(self):
        pose = np.array([[self.x], [self.y], [self.theta]])
        return pose

    def setCovariance(self, new_Cov):
        self.posCov = new_Cov

    def getCovariance(self):
        return self.poseCovariance

    def move(self, robotCurrentAbs, u):
        [nextRobotAbs, H1, H2] = Relative2AbsolutePose(robotCurrentAbs, u)
        self.x = nextRobotAbs[0][0]
        self.y = nextRobotAbs[1][0]
        self.theta = nextRobotAbs[2][0]
        return nextRobotAbs, H1, H2

    def sense(self, robotCurrentAbs, landmarkAbs):
        if self.senseType == 'Vision':
            [measurement, H1, H2] = Absolute2RelativeXY(robotCurrentAbs, landmarkAbs)

        else:
            raise ValueError('Unknown Measurement Type')

        return measurement, H1, H2

    def inverseSense(self, robotCurrentAbs, measurement):

        if self.senseType == 'Vision':
            [landmarkAbs, H1, H2] = Relative2AbsoluteXY(robotCurrentAbs, measurement)

        else:
            raise ValueError('Unknown Measurement Type')

        return landmarkAbs, H1, H2


class LandmarkMeasurement():

    def __init__(self, meas_Cov):
        self.measurementCovariance = meas_Cov

    def setCovariance(self, new_Cov):
        self.measurementCovariance = new_Cov

    def getCovariance(self):
        return self.measurementCovariance


class Motion():

    def __init__(self, motion_command, motion_Cov):
        self.u = motion_command
        self.motionCovariance = motion_Cov

    def setCovariance(self, new_Cov):
        self.motionCovariance = new_Cov

    def getCovariance(self):
        return self.motionCovariance

    def setMotionCommand(self, motionCommand):
        self.u = motionCommand

    def getMotionCommand(self):
        return self.u


class KalmanFilter():

    def __init__(self, mean, covariance, robot):

        self.stateMean = mean
        self.stateCovariance = covariance
        self.robot = robot

    def setStateMean(self, mean):

        self.stateMean = mean

    def getStateMean(self):

        return self.stateMean

    def setStateCovariance(self, covariance):

        self.stateCovariance = covariance

    def getStateCovariance(self):

        return self.stateCovariance

    def predict(self, motion, motionCovariance):
        # TODO get robot current pose
        pose = self.robot.getPose()

        # TODO move robot given current pose and u
        [next_robot_abs, F, W] = self.robot.move(pose, motion)

        # TODO predict state mean
        priorStateMean = self.getStateMean()
        priorStateMean[0:3] = next_robot_abs[0:3]

        # TODO predict state covariance
        robot_cov = self.robot.getCovariance()
        priorRobotCovariance = np.dot(np.dot(F, robot_cov), np.transpose(F)) + np.dot(np.dot(W, motionCovariance), np.transpose(W))# + np.dot(np.identity(3),0.15)*np.sqrt(motion[0][0]*motion[0][0]+motion[1][0]*motion[1][0])

        # TODO set robot new pose
        self.robot.setPose(next_robot_abs)

        # TODO set robot new covariance
        self.robot.poseCovariance = priorRobotCovariance

        # TODO set KF priorStateMean
        self.setStateMean(priorStateMean)

        # TODO set KF priorStateCovariance
        statecov = np.array(self.getStateCovariance())
        statecov[0:3][:,:3] = np.dot(priorRobotCovariance[0:3][:,:3], 1)
        self.setStateCovariance(statecov)

    def update(self, measurement, measurementCovariance, new):
        global seenLandmarks_
        global dimR_
        # TODO get robot current pose
        u = self.robot.getPose()
        # get landmark absolute position estimate given current pose and measurement (robot.sense)
        [landmarkAbs, G1, G2] = self.robot.inverseSense(u, measurement)

        # TODO get KF state mean and covariance
        stateMean = self.getStateMean()
        stateCovariance = self.getStateCovariance()

        # if new landmark augment stateMean and stateCovariance
        if new:
            stateMean = np.concatenate((stateMean, [[landmarkAbs[0]], [landmarkAbs[1]]]), axis=0)
            Prr = self.robot.getCovariance()
            if len(seenLandmarks_) == 1:
                Plx = np.dot(G1, Prr)
            else:
                lastStateCovariance = KalmanFilter.getStateCovariance(self)
                Prm = lastStateCovariance[0:3, 3:]
                Plx = np.dot(G1, np.bmat([[Prr, Prm]]))
            Pll = np.array(np.dot(np.dot(G1, Prr), np.transpose(G1))) + np.array(
                np.dot(np.dot(G2, measurementCovariance), np.transpose(G2)))
            P = np.bmat([[stateCovariance, np.transpose(Plx)], [Plx, Pll]])
            stateCovariance = P
            # TODO: set state covariance
            self.setStateCovariance(stateCovariance)

            # TODO: set state mean
            self.setStateMean(stateMean)
        else:
            # if old landmark stateMean & stateCovariance remain the same (will be changed in the update phase by the kalman gain)
            # TODO calculate expected measurement
            pos = [i for i, x in enumerate(seenLandmarks_) if x == measurement[2]]
            # sense calls Absolute2RelativeXY which given robot pose and landmark abs coordinates
            # returns expected measurement from robot
            [expectedMeasurement, Hr, Hl] = self.robot.sense(u,stateMean[3+pos[0]*2:5+pos[0]*2])

            # get measurement
            Z = ([[measurement[0]], [measurement[1]]])
            # Update
            x = stateMean
            label = measurement[2]
            # TODO y = Z - expectedMeasurement
            # y is the innovation
            y = np.subtract(Z, expectedMeasurement)

            # build H, also known as C
            # H = [Hr, 0, ..., 0, Hl,  0, ..,0] position of Hl depends on when was the landmark seen?
            H = np.reshape(Hr, (2, 3))
            for i in range(0, seenLandmarks_.index(label)):
                H = np.bmat([[H, np.zeros([2, 2])]])
            H = np.bmat([[H, np.reshape(Hl, (2, 2))]])
            for i in range(0, len(seenLandmarks_) - seenLandmarks_.index(label) - 1):
                H = np.bmat([[H, np.zeros([2, 2])]])
            # TODO compute S - landmark covariance - innovation covariance
            S = np.dot(np.dot(H, stateCovariance), np.transpose(H)) + measurementCovariance

            #mahalanobis distance
            MHdist = np.dot(np.dot(np.transpose(y), np.linalg.pinv(S)),y)

            if (S < 0.0001).all():
                print('Non-invertible S Matrix')
                raise ValueError
                return
            # reject updates with high mahalanobis distance
            #elif MHdist[0][0] > 9.218:

            elif MHdist[0][0] > 7.378:
                print('Too far away rejecting')
                return
            else:
                # TODO calculate Kalman gain
                K = np.matmul(np.matmul(stateCovariance, H.transpose()), np.linalg.pinv(S))

                # TODO compute posterior mean
                posteriorStateMean = x + np.matmul(K, y)

                # TODO compute posterior covariance
                posteriorStateCovariance = stateCovariance - np.matmul(np.matmul(K, H), stateCovariance)

                # check theta robot is a valid theta in the range [-pi, pi]
                posteriorStateMean[2][0] = pi2pi(posteriorStateMean[2][0])
                # TODO update robot pose
                robotPose = np.array(posteriorStateMean[0:3])

                # set robot pose
                self.robot.setPose(robotPose)
                # TODO updated robot covariance
                # get robot covaiance
                robotCovariance = posteriorStateCovariance[0:3][:, :3]
                # set robot covariance
                self.robot.poseCovariance = np.array(robotCovariance)

                # set posterior state mean
                KalmanFilter.setStateMean(self, posteriorStateMean)

                # set posterior state covariance
                KalmanFilter.setStateCovariance(self, posteriorStateCovariance)


class SLAM():
    def __init__(self):
        self.time = rospy.get_time()
        self.IMU_time = rospy.get_time()
        self.IMU_th = 0
        self.count = 1
        self.id = 0
        self.data_list = []

        # Initialise robot
        # TODO initialise a robot pose and covariance
        robot_pose = [[0], [0], [0]]
        robot_covariance = [[0.0, 0.00,0.00],[0.00,0.00,0.00],[0.0,0.00,0.00]]
        sense_Type = 'Vision'
        self.robot = Robot(robot_pose, robot_covariance, sense_Type)

        # Initialise motion
        # TODO initialise a motion command and covariance
        motion_command = [0, 0, 0]
        motion_covariance = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.motion = Motion(motion_command, motion_covariance)

        # Initialise landmark measurement
        # TODO initialise a measurement covariance
        measurement_covariance = [[0, 0], [0, 0]]
        self.landmarkMeasurement = LandmarkMeasurement(measurement_covariance)

        # Initialise kalman filter
        # TODO initialise a state mean and covariance
        state_mean = [0,0,0]
        state_covariance = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # initial state contains initial robot pose and covariance
        self.KF = KalmanFilter(state_mean, state_covariance, self.robot)

        # TODO Subscribe to different topics and assign their respective callback functions
        IMUsub = message_filters.Subscriber("/mobile_base/sensors/imu_data",Imu)
        odomSub = message_filters.Subscriber("/odom", Odometry)
        cylSub = message_filters.Subscriber("/cylinderTopic", cylDataArray)
        ts = message_filters.ApproximateTimeSynchronizer([odomSub, cylSub, IMUsub], queue_size=150, slop=0.07)
        ts.registerCallback(self.callback)
        #rospy.spin()

    # receives synchronized messages, calls other callback functions when received
    def callback(self, odom_msg, cyl_msg, IMU_msg):
        self.id = self.id + 1
        self.callbackOdometryMotion(odom_msg)
        self.callbackLandmarkMeasurement(cyl_msg)
        self.callbackIMU(IMU_msg)

    # gyro callback function to update IMU_th class attribute
    def callbackIMU(self, msg):
        t_new = rospy.get_time()
        t_old = self.IMU_time
        dt = (t_new - t_old)
        self.IMU_time = t_new
        if dt > 1:
            return
        self.IMU_th = msg.angular_velocity.z * dt
        # filter results
        if self.IMU_th > 1.05:
            print("changing imu")
            self.IMU_th = 1
        if self.IMU_th < -1.05:
            print("changing imu")
            self.IMU_th = -1

    # receives odom message and call predict
    def callbackOdometryMotion(self, msg):
        # TODO read msg received
        xdot = msg.twist.twist.linear.x * 1.064
        ydot = msg.twist.twist.linear.y
        thdot_odom = msg.twist.twist.angular.z

        # filter results
        '''if xdot > 0.22:
            print(xdot)
            print("changing xdot")
            xdot = 0.2
        if thdot_odom > 1.1:
            print("changing thdot")
            xdot = 1
        if thdot_odom < -1.1:
            print("changing thdot")
            xdot = -1'''

        # TODO You can choose to only rely on odometry or read a second sensor measurement
        #th_IMU = self.IMU_th * 0.28 * 0.85
        th_IMU = self.IMU_th * 0.28 * 1.05 #* 2.5

        # TODO compute dt = duration of the sensor measurement
        t_new = rospy.get_time()
        t_old = self.time
        dt = (t_new - t_old)
        self.time = t_new

        # use gyro measurement if there is a notable difference between odom and gyro angular velocities
        #if abs(th_IMU - thdot_odom*dt) > (0.03*(math.pi/180))/dt:
        if abs(th_IMU - thdot_odom * dt) > (0.05 * (math.pi / 180)) / dt:
            dth = th_IMU
        else:
            dth = thdot_odom*dt*0.45

        # if robot is moving forward use gyro
        if (xdot != 0):
            dth = th_IMU
        #dth = thdot_odom*dt*0.45
        #dth = th_IMU*0.85
        dth = dth*1.95
        #dth = dth *1.2


        # TODO compute command
        dx = xdot * dt * 0.48
        dy = ydot * dt * 0.48
        u = [[dx], [dy], [dth]]
        pose_prior = self.robot.getPose()
        [pose_after,H1,H2] = self.robot.move(pose_prior,u)
        self.data_list.append(["POSE2D",self.id, pose_after[0][0],pose_after[1][0], pose_after[2][0]])

        innov = np.subtract(pose_after,pose_prior)
        # print('Innovation:',innov)
        Mahadist = np.dot(np.dot(np.transpose(innov),np.linalg.pinv(self.motion.getCovariance())),innov)
        if float(Mahadist) > 1000 & self.count >100:
            print("try turn too far")
            self.count += 1
            return
        else:
            self.count += 1


        # set motion command
        self.motion.setMotionCommand(u)
        # get covariance from msg received
        covariance = msg.twist.covariance
        #self.motion.setCovariance([[covariance[0], 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, covariance[35]]])
        #defining norm of velocity
        velnorm = math.sqrt((dx)**2+(dy)**2+(dth)**2)
        print(velnorm)
        #print(velnorm)
        #self.motion.setCovariance([[0.005, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0005]])

        self.motion.setCovariance(np.dot(velnorm,[[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0,0.2]]))
        #print("motion cov: ", self.motion.getCovariance())
        #poseCovariance = self.robot.getCovariance() # Rowan - what's this for????

        # call KF to execute a prediction
        self.KF.predict(self.motion.getMotionCommand(), self.motion.getCovariance())

    def callbackLandmarkMeasurement(self, data):
        global seenLandmarks_
        for i in range(0, len(data.cylinders)):
            # read data received
            # aligning landmark measurement frame with robot frame
            dx = data.cylinders[i].Zrobot #- 0.04
            dy = -data.cylinders[i].Xrobot
            # account for cylinder offset
            angle = math.atan(dy/dx)
            hypo = np.hypot(dx, dy) + 0.055
            dx = hypo*math.cos(angle)
            dy = hypo*math.sin(angle)

            # filter results
            if dx > 2.6 or dx < 0.6 or dy < -2.5 or dy > 2.5:
                continue

            label = data.cylinders[i].label
            # determine if landmark is seen for first time
            # or it's a measurement of a previously seen landamrk
            new = 0
            # if seenLandmarks_ is empty
            if not seenLandmarks_:
                new = 1
                seenLandmarks_.append(label)
            # if landmark was seen previously
            elif label not in seenLandmarks_:
                new = 1
                seenLandmarks_.append(label)
            measurement = []
            measurement.append(dx)
            measurement.append(dy)
            measurement.append(label)


            [sig_xx_sqared, sig_xy_sqared, sig, sig_yy_sqared] = data.cylinders[i].covariance
            [landm_dims, h1, h2] = Relative2AbsoluteXY(self.robot.getPose(), [dx, dy])
            cyl_info = [label, landm_dims[0], landm_dims[1]]

            #self.data_list.append(["LANDMARK_MEAS2D", self.id-1, self.id, dx, dy,
            #                  sig_xx_sqared, sig_xy_sqared, sig_yy_sqared])
            self.data_list.append(["POINT2D", self.id, landm_dims[0], landm_dims[1]])



            # depth error
            sig = 0.00471429*dx**2 - 0.0106429*dx + 0.00714288
            self.landmarkMeasurement.setCovariance([[sig, 0.0], [0.0, sig]])
            measurementLandmarkCovariance = self.landmarkMeasurement.getCovariance()
            measurementLandmarkCovariance = np.dot(measurementLandmarkCovariance,2)

            # call KF to execute an update
            #try:
                #self.KF.update(measurement, measurementLandmarkCovariance, new)
            #except ValueError:
            #    return

# return any keystroke pressed on the keyboard
def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

if __name__ == '__main__':
    print('Landmark SLAM Started...')
    print('jamunji')
    settings = termios.tcgetattr(sys.stdin)

    # Initialize the node and name it.
    rospy.init_node('listener')
    pub_pose = rospy.Publisher('/odom_new', Odometry,queue_size=1)
    pub_marker =  rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=50)

    markerArray = MarkerArray()
    # Go to class functions that do all the heavy lifting. Do error checking.
    t0 = t.time()
    try:
        slam = SLAM()
    except rospy.ROSInterruptException:
        pass
    while True:
        if t.time()-t0 > 110:
            with open('/home/turtlebot/catkin_ws/src/pack/src/data.csv', 'wb') as csvfile:
                writer = csv.writer(csvfile)#, delimiter=' ')
                #writer = csv.writer(f, delimiter='\t', lineterminator='\n')
                for line in slam.data_list:

                    writer.writerow(line)


            break

            #print("state mean: ", slam.KF.stateMean)
            '''with open('/home/turtlebot/catkin_ws/src/pack/src/solution.csv', 'wb') as csvfile:
                writer = csv.writer(csvfile)#, delimiter=' ')
                for i in range(0,9):
                    writer.writerow(["POINT2D", seenLandmarks_[i], np.array(slam.KF.stateMean[3+i*2])[0][0], np.array(slam.KF.stateMean[4+i*2])[0][0]])

            print(ErrorFunction('/home/turtlebot/catkin_ws/src/pack/src/solution.csv','/home/turtlebot/catkin_ws/src/pack/src/gt.csv'))
            break'''
        key = getKey()

        if (key == '\x03'):
            break

        pose = slam.robot.getPose()


        x0 = pose[0][0]
        y0 = pose[1][0]
        theta0 = pose[2][0]

        # convert to quaternion dims
        quaternion = quaternion_from_euler(0, 0, theta0)
        robot_cov = slam.robot.getCovariance()
        #print("robot cov",robot_cov)
        cov = [0]*36
        cov[0] = robot_cov[0][0]
        #cov[1] = robot_cov[0][0]
        cov[7] = robot_cov[1][1]
        #scov[35] = robot_cov[2][2]
        #print("cov: ",cov)
        #print(cov)
        #print(slam.robot.getPose())

        #print(x0)
        # create object, give it values then publish to new topic
        odom_object = Odometry()
        odom_object.pose.pose.position.x = x0
        odom_object.pose.pose.position.y = y0
        odom_object.pose.pose.orientation.x = quaternion[0]
        odom_object.pose.pose.orientation.y = quaternion[1]
        odom_object.pose.pose.orientation.z = quaternion[2]
        odom_object.pose.pose.orientation.w = quaternion[3]
        odom_object.pose.covariance = cov
        odom_object.header.frame_id = "base_link"
        odom_object.child_frame_id = "base_footprint"
        odom_object.header.stamp = rospy.Time.now()
        odom_object.header.seq=odom_object.header.seq+1

        pub_pose.publish(odom_object)

        #marker plotting
        size = len(slam.KF.stateMean) - 3
        # print(size)
        j = 0
        for i in range(size):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time()
            marker.type = marker.CYLINDER
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.b = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = slam.KF.stateMean[3+j][0]
            marker.pose.position.y = slam.KF.stateMean[3+j+1][0]
            marker.pose.position.z = 0.0
            marker.id = i


            markerArray.markers.append(marker)


            if size == j+2:
                break
            else:
                j = j+2
        #print('hello')
        pub_marker.publish(markerArray)

        #ground truth.
        gt =[0,-0.5,1.5,-0.5,2.5,0,2.5,2,2,3,1,1,0,3,-0.5,2,1.5,0.5]
        k = 1
        j = 0
        if k == 1:
            for i in range(len(gt)):
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.header.stamp = rospy.Time()
                marker.type = marker.CYLINDER
                marker.action = marker.ADD
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
                marker.color.a = 1.0
                marker.color.b = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.5
                marker.color.b = 0.5
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.pose.position.x = gt[j]
                marker.pose.position.y = gt[j+1]
                marker.pose.position.z = 0.0
                marker.id = i+10

                markerArray.markers.append(marker)

                if len(gt) == j+2:
                    break
                else:
                    j = j+2
                pub_marker.publish(markerArray)