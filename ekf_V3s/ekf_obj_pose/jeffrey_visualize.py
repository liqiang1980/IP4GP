# import numpy as np
# import matplotlib.pyplot as plt

# # Load data from text files
# x_bar_all = np.loadtxt('offline_x_bar_all.txt')
# x_state_all = np.loadtxt('offline_x_state_all.txt')
# gd_all = np.loadtxt('offline_x_gt_palm.txt')

# # Plot the results
# plt.figure()

# # Plot x, y, and z positions
# for i in range(7):
#     plt.subplot(7, 1, i + 1)
#     # plt.plot(x_bar_all[:, i], label='x_bar')
#     plt.plot(x_state_all[:, i], label='predict')
#     plt.plot(gd_all[:, i], label='GT')
#     plt.legend('upper left')

# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Load data from text files
x_bar_all = np.loadtxt('offline_x_bar_all.txt')
x_state_all = np.loadtxt('offline_x_state_all.txt')
gd_all = np.loadtxt('offline_x_gt_palm.txt')

# Plot the results
plt.figure(figsize=(10, 20))

# Define labels for each subplot
labels = ['x position', 'y position', 'z position', 'x rotation', 'y rotation', 'z rotation', 'w']

# Plot x, y, and z positions
for i in range(7):
    plt.subplot(7, 1, i + 1)
    plt.plot(x_bar_all[:, i], color='y', alpha=1,label='x_bar')
    plt.plot(x_state_all[:, i], color='r', alpha=0.5,label='predict')
    plt.plot(gd_all[:, i], color='b', alpha=0.5,label='GT')
    plt.legend(loc='upper left')
    # plt.xlabel('Time step')
    plt.ylabel(labels[i])
    # plt.title(labels[i] + ' vs Time step')

plt.tight_layout()
plt.show()




def interaction(self, object_param, ekf_grasping, tacp, basicData):
    """
    Do one EKF prediction round
    """
    print("Round ", self.cnt_test, "......................")
    self.cnt_test += 1
    """ Update Joint state and FK """
    self.fk.fk_update_all(basicData=basicData)
    """ 
    Update contact state:
    1. contact_flags
    2. number of contact finger parts
    3. cur_contact_tac state 
    """
    # print(">>>>time 0")
    tacp.is_fingers_contact(basicData=basicData, f_param=self.f_param, fk=self.fk)
    # print(">>>>time 1")

    """ Update gd_state """
    gd_state = basicData.obj_palm_posrotvec
    # print("gd:  ", gd_state)

    """ First interaction, Do Initialization """
    # print("At beginning, xstate: ", self.x_state)
    if not self.FIRST_INTERACTION_FLAG:
        self.FIRST_INTERACTION_FLAG = True
        """ x_state Initialization """
        self.x_state[:6] = gd_state
        np.set_printoptions(suppress=True)
        # print('x_state from beginning before add noise', self.x_state)
        if tacCONST.initE_FLAG:
            init_e = np.hstack(
                (np.random.uniform((-1) * float(object_param[1]), float(object_param[1]), (1, 3)),
                 np.random.uniform(-1 * float(object_param[2]), float(object_param[2]), (1, 3))))
            self.x_state[:6] = np.ravel(self.x_state[:6] + init_e)
        self.Xcomponents_update(x=self.x_state[:6])  # Mathematical components initialization

        """ 
        Augmented state Initialization.
        augmented state with the contact position on the object surface described in the object frame 
        """
        self.x_state = self.augmented_state(basicData=basicData, tacp=tacp, xstate=self.x_state)
        # print("  x_state check: ", self.x_state, "\n", np.ravel(self.x_state)[3:6])

        """ Init the data for plot """
        x_state_plot = [0., 0., 0., 0., 0., 0., 0.]
        x_bar_plot = [0., 0., 0., 0., 0., 0., 0.]
        gd_state_plot = [0., 0., 0., 0., 0., 0., 0.]
        x_state_plot[0:3] = np.ravel(self.x_state)[0:3]
        x_state_plot[3:6], x_state_plot[6] = ug.normalize_scale(np.ravel(self.x_state)[3:6])
        x_bar_plot[0:3] = np.ravel(self.x_state)[0:3]
        x_bar_plot[3:6], x_bar_plot[6] = ug.normalize_scale(np.ravel(self.x_state)[3:6])
        gd_state_plot[0:3] = gd_state[0:3]
        gd_state_plot[3:6], gd_state_plot[6] = ug.normalize_scale(gd_state[3:6])
        self.x_bar_all = np.vstack((self.x_bar_all, x_bar_plot))
        self.x_state_all = np.vstack((self.x_state_all, x_state_plot))
        self.gd_all = np.vstack((self.gd_all, gd_state_plot))

        """Set first contact flags for finger parts"""
        for f_part in self.f_param:
            f_name = f_part[0]
            if tacp.is_contact[f_name]:
                tacp.is_first_contact[f_name] = True
        """ Initialization Done """
        tacp.Last_tac_renew(f_param=self.f_param)
        self.Xcomponents_update(x=self.x_state[:6])  # Mathematical components update for next EKF round
        print('\n...................Initialization done...................\n')
        return

    """ Detect new contact tacs that have never been touched before """
    if self.cnt_test < 10:
        for idx, f_part in enumerate(self.f_param):
            f_name = f_part[0]
            if tacp.is_contact[f_name] and not tacp.is_first_contact[f_name]:
                self.x_state = self.update_augmented_state(idx=idx, f_name=f_name,
                                                           tacp=tacp, xstate=self.x_state)
                tacp.is_first_contact[f_name] = True
    """ If contact, always contact """
    tacp.is_contact = deepcopy(tacp.is_first_contact)  # This code overrides the previous renew of tacp.is_contact
    # print("contact:", tacp.is_contact)
    # print("is_first_contact:", tacp.is_first_contact)

    """ EKF Forward prediction """
    self.x_bar, P_state_cov = ekf_grasping.state_predictor(xstate=self.x_state,
                                                           P_state_cov=self.P_state_cov,
                                                           tacp=tacp,
                                                           robctrl=self)
    # self.x_bar = deepcopy(_x_bar)
    # Mathematical components update by result of Forward prediction for Posteriori estimation
    self.Xcomponents_update(x=self.x_bar[:6])

    """ h_t & z_t updates """
    h_t_position, h_t_nv = ekf_grasping.observe_computation(tacp=tacp, robctrl=self)
    z_t_position, z_t_nv = ekf_grasping.measure_fb(tacp=tacp, robctrl=self)
    if tacCONST.PN_FLAG == 'p':
        self.z_t = np.ravel(z_t_position)
        self.h_t = np.ravel(h_t_position)
    else:
        self.z_t = np.concatenate((z_t_position, z_t_nv), axis=None)
        self.h_t = np.concatenate((h_t_position, h_t_nv), axis=None)

    """ EKF Posteriori estimation """
    if tacCONST.posteriori_FLAG:
        self.x_state, self.P_state_cov = ekf_grasping.ekf_posteriori(x_bar=self.x_bar,
                                                                     z_t=self.z_t,
                                                                     h_t=self.h_t,
                                                                     P_state_cov=P_state_cov,
                                                                     tacp=tacp,
                                                                     robctrl=self)
    else:
        self.x_state = self.x_bar

    """ Update the plot_data """
    x_state_plot = [0., 0., 0., 0., 0., 0., 0.]
    x_bar_plot = [0., 0., 0., 0., 0., 0., 0.]
    gd_state_plot = [0., 0., 0., 0., 0., 0., 0.]
    x_state_plot[0:3] = self.x_state[0:3]
    x_state_plot[3:6], x_state_plot[6] = ug.normalize_scale(self.x_state[3:6])
    x_bar_plot[0:3] = self.x_bar[0:3]
    x_bar_plot[3:6], x_bar_plot[6] = ug.normalize_scale(self.x_bar[3:6])
    gd_state_plot[0:3] = gd_state[0:3]
    gd_state_plot[3:6], gd_state_plot[6] = ug.normalize_scale(gd_state[3:6])
    self.x_bar_all = np.vstack((self.x_bar_all, x_bar_plot))
    self.gd_all = np.vstack((self.gd_all, gd_state_plot))
    self.x_state_all = np.vstack((self.x_state_all, x_state_plot))
    #
    np.savetxt('offline_x_bar_all.txt', self.x_bar_all)
    np.savetxt('offline_x_state_all.txt', self.x_state_all)
    np.savetxt('offline_x_gt_palm.txt', self.gd_all)
    """ Last Update:
    Update last_state by cur_state.
    Update mathematical components.
    """
    tacp.Last_tac_renew(f_param=self.f_param)
    self.Xcomponents_update(x=self.x_state[:6])  # Mathematical components update for next EKF round
    print(".........................................")