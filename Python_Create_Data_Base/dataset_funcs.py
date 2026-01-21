import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import os


def fillline(startp, endp, pts):
    """
    Take starting, ending point & number of points in between.
    Make line to connect between 2 coordinates.
    
    Parameters
    ----------
    startp : array-like
        Starting point [x, y]
    endp : array-like
        Ending point [x, y]
    pts : int
        Number of points in the line
    
    Returns
    -------
    xx : numpy.ndarray
        X coordinates of the line
    yy : numpy.ndarray
        Y coordinates of the line
    """
    # Calculate gradient
    m = (endp[1] - startp[1]) / (endp[0] - startp[0]) if (endp[0] - startp[0]) != 0 else np.inf

    if m == np.inf:  # vertical line
        xx = np.full(pts, startp[0])
        yy = np.linspace(startp[1], endp[1], pts)
    elif m == 0:  # horizontal line
        xx = np.linspace(startp[0], endp[0], pts)
        yy = np.full(pts, startp[1])
    else:  # regular line
        xx = np.linspace(startp[0], endp[0], pts)
        yy = m * (xx - startp[0]) + startp[1]

    return xx, yy


def plotcube(edges=None, origin=None, alpha=None, color=None, ax=None):
    """
    PLOTCUBE - Display a 3D-cube in the current axes
    
    Parameters
    ----------
    edges : array-like, optional
        3-elements vector that defines the length of cube edges.
        Default: [10, 56, 100]
    origin : array-like, optional
        3-elements vector that defines the start point of the cube.
        Default: [10, 10, 10]
    alpha : float, optional
        Scalar that defines the transparency of the cube faces (from 0 to 1).
        Default: 0.7
    color : array-like, optional
        3-elements vector that defines the faces color of the cube.
        Default: [1, 0, 0] (red)
    ax : matplotlib 3D axes, optional
        The axes to plot on. If None, uses current axes.
    
    Returns
    -------
    ax : matplotlib 3D axes
        The axes with the plotted cube
    
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> plotcube([5, 5, 5], [2, 2, 2], 0.8, [1, 0, 0], ax)
    >>> plotcube([5, 5, 5], [10, 10, 10], 0.8, [0, 1, 0], ax)
    >>> plotcube([5, 5, 5], [20, 20, 20], 0.8, [0, 0, 1], ax)
    >>> plt.show()
    """
    
    # Default input arguments
    if edges is None:
        edges = [10, 56, 100]
    if origin is None:
        origin = [10, 10, 10]
    if alpha is None:
        alpha = 0.7
    if color is None:
        color = [1, 0, 0]
    
    edges = np.array(edges)
    origin = np.array(origin)
    color = np.array(color)
    
    # Get or create axes
    if ax is None:
        ax = plt.gca()
    
    # Define the 6 faces of the cube
    # Each face is defined by 4 vertices (x, y, z coordinates)
    XYZ = [
        # Face 1 (x = 0)
        [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 0]],
        # Face 2 (x = 1)
        [[1, 1, 1, 1], [0, 0, 1, 1], [0, 1, 1, 0]],
        # Face 3 (y = 0)
        [[0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 1, 1]],
        # Face 4 (y = 1)
        [[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]],
        # Face 5 (z = 0)
        [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
        # Face 6 (z = 1)
        [[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]],
    ]
    
    # Scale and translate each face
    faces = []
    for face in XYZ:
        # Scale by edges and translate by origin
        x = np.array(face[0]) * edges[0] + origin[0]
        y = np.array(face[1]) * edges[1] + origin[1]
        z = np.array(face[2]) * edges[2] + origin[2]
        # Create vertices for this face (4 corners)
        verts = list(zip(x, y, z))
        faces.append(verts)
    
    # Create 3D polygon collection
    poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='k', linewidth=0.5)
    ax.add_collection3d(poly3d)
    
    return ax


def create_locations_18_dynamic(L, R, noise_R, num_jumps):
    """
    Create dynamic speaker locations within a room for microphone array simulation.
    
    Parameters
    ----------
    L : array-like
        Room dimensions [room_x, room_y, room_z]
    R : float
        Radius of the speaker circle around microphone array
    noise_R : float
        Noise radius for position perturbation
    num_jumps : int
        Number of speaker position jumps/locations to generate
    
    Returns
    -------
    s_first : numpy.ndarray
        First speaker positions (3 x num_jumps) [x; y; z]
    label_first : numpy.ndarray
        Angular labels for first speaker positions
    s_second : numpy.ndarray
        Second speaker positions (3 x num_jumps) [x; y; z]
    label_second : numpy.ndarray
        Angular labels for second speaker positions
    s_noise : numpy.ndarray
        Noise source position [x, y, z]
    r : numpy.ndarray
        Microphone array positions (4 x 3)
    """
    
    # %% variable
    room_x = L[0]
    room_y = L[1]
    high = 1
    angle = 360
    radius_mics = 0.1
    distance_from_woll = 0.5
    
    # %% create circle & line & mic location
    distance_total = R + distance_from_woll + noise_R
    end_point_x = room_x - (R + distance_from_woll + noise_R)
    end_point_y = room_y - (R + distance_from_woll + noise_R)
    Radius_X = (end_point_x - distance_total) * np.random.rand() + distance_total
    Radius_Y = (end_point_y - distance_total) * np.random.rand() + distance_total
    
    # take rand 180 degrees of circle to create rand orientation of Microphone array
    R_angle = np.random.randint(1, angle + 1)  # randi([1,angle])
    t = np.linspace(-np.pi, 2 * np.pi, angle + angle // 2)  # 3 times 180 to create option to rand whole circ
    # MATLAB: t=t(R_angle:R_angle+angle/2-1) -> Python: 0-based indexing
    t = t[R_angle - 1:R_angle - 1 + angle // 2]
    x = R * np.sin(t) + Radius_X
    y = R * np.cos(t) + Radius_Y
    
    circ_mics_x = radius_mics * np.sin(t) + Radius_X
    circ_mics_y = radius_mics * np.cos(t) + Radius_Y
    line_x, line_y = fillline([x[0], y[0]], [x[angle // 2 - 1], y[angle // 2 - 1]], int(R * 2 * 100))
    
    # r= [circ_mics_x(1) circ_mics_y(1) high; circ_mics_x(25) circ_mics_y(25) high; circ_mics_x(55) circ_mics_y(55) high; circ_mics_x(80) circ_mics_y(80) high;...
    #     circ_mics_x(120) circ_mics_y(120) high; circ_mics_x(150) circ_mics_y(150) high; circ_mics_x(180) circ_mics_y(180) high];
    
    # MATLAB indices 1, 55, 120, 180 -> Python indices 0, 54, 119, 179
    r = np.array([
        [circ_mics_x[0], circ_mics_y[0], high],
        [circ_mics_x[54], circ_mics_y[54], high],
        [circ_mics_x[119], circ_mics_y[119], high],
        [circ_mics_x[179], circ_mics_y[179], high]
    ])
    
    # %% create x1 y1 x2 y2
    center = np.array([Radius_X, Radius_Y])
    start_circ_vec = np.array([line_x[0], line_y[0]]) - center
    labels_location = np.arange(5, 176, 10)  # 5:10:175 in MATLAB
    list_locations = []
    
    # Initialize arrays
    x1 = np.zeros(num_jumps)
    y1 = np.zeros(num_jumps)
    x2 = np.zeros(num_jumps)
    y2 = np.zeros(num_jumps)
    x1_temp = np.zeros(num_jumps)
    y1_temp = np.zeros(num_jumps)
    x2_temp = np.zeros(num_jumps)
    y2_temp = np.zeros(num_jumps)
    label_first = np.zeros(num_jumps, dtype=int)
    label_second = np.zeros(num_jumps, dtype=int)
    
    i = 0  # Python 0-based indexing (was i=1 in MATLAB)
    while i < num_jumps:
        next_speech = True
        number_of_loops = 0
        while next_speech:
            if number_of_loops > 300:
                i = 0
                list_locations = []
            number_of_loops += 1
            
            rand1 = np.random.randint(0, angle // 2)  # randi(angle/2) -> 0 to angle/2-1
            x1[i] = x[rand1]
            y1[i] = y[rand1]
            x1_temp[i] = x1[i]
            y1_temp[i] = y1[i]
            w = 0.01 * np.random.randint(1, 315)  # randi([1,314])
            x1[i] = x1[i] + noise_R * np.sin(w)
            y1[i] = y1[i] + noise_R * np.cos(w)
            first_vec = np.array([x1[i], y1[i]]) - center
            ang1 = np.degrees(np.arccos(
                (start_circ_vec[0] * first_vec[0] + start_circ_vec[1] * first_vec[1]) /
                (np.linalg.norm(start_circ_vec) * np.linalg.norm(first_vec))
            ))
            label_first[i] = np.argmin(np.abs(labels_location - ang1))
            
            rand2 = np.random.randint(0, angle // 2)
            x2[i] = x[rand2]
            y2[i] = y[rand2]
            x2_temp[i] = x2[i]
            y2_temp[i] = y2[i]
            w = 0.01 * np.random.randint(1, 315)
            x2[i] = x2[i] + noise_R * np.sin(w)
            y2[i] = y2[i] + noise_R * np.cos(w)
            second_vec = np.array([x2[i], y2[i]]) - center
            ang2 = np.degrees(np.arccos(
                (start_circ_vec[0] * second_vec[0] + start_circ_vec[1] * second_vec[1]) /
                (np.linalg.norm(start_circ_vec) * np.linalg.norm(second_vec))
            ))
            label_second[i] = np.argmin(np.abs(labels_location - ang2))
            
            if label_first[i] in list_locations or label_second[i] in list_locations:
                next_speech = True
            else:
                next_speech = False
            
            loc_xy = np.array([[x1[i], y1[i]], [x2[i], y2[i]]])
            dist = pdist(loc_xy, 'euclidean')[0]
            if dist < 0.5:
                next_speech = True
            
            if i > 0:
                loc_xy1 = np.array([[x1[i], y1[i]], [x2[i - 1], y2[i - 1]]])
                loc_xy2 = np.array([[x1[i - 1], y1[i - 1]], [x2[i], y2[i]]])
                dist_last1 = pdist(loc_xy1, 'euclidean')[0]
                dist_last2 = pdist(loc_xy2, 'euclidean')[0]
                if dist_last1 < 0.5 or dist_last2 < 0.5:
                    next_speech = True
        
        list_locations.append(label_first[i])
        list_locations.append(label_second[i])
        i += 1
    
    # %% create location of the speakers
    s_first = np.array([x1, y1, np.ones(num_jumps)])
    s_second = np.array([x2, y2, np.ones(num_jumps)])
    
    # %% create location (recalculate labels)
    labels_location = np.arange(5, 176, 10)
    for i in range(num_jumps):
        center = np.array([Radius_X, Radius_Y])
        start_circ_vec = np.array([line_x[0], line_y[0]]) - center
        first_vec = np.array([x1[i], y1[i]]) - center
        second_vec = np.array([x2[i], y2[i]]) - center
        ang1 = np.degrees(np.arccos(
            (start_circ_vec[0] * first_vec[0] + start_circ_vec[1] * first_vec[1]) /
            (np.linalg.norm(start_circ_vec) * np.linalg.norm(first_vec))
        ))
        ang2 = np.degrees(np.arccos(
            (start_circ_vec[0] * second_vec[0] + start_circ_vec[1] * second_vec[1]) /
            (np.linalg.norm(start_circ_vec) * np.linalg.norm(second_vec))
        ))
        label_first[i] = np.argmin(np.abs(labels_location - ang1))
        label_second[i] = np.argmin(np.abs(labels_location - ang2))
    
    # %% noise
    middle = np.array([Radius_X, Radius_Y, high])
    s_noise = np.array([Radius_X, Radius_Y, high])
    d_noise = np.linalg.norm(s_noise - middle)
    while d_noise < 2:
        x_noise = distance_from_woll + 0.01 * np.random.randint(1, 101) * (room_x - 2 * distance_from_woll)
        y_noise = distance_from_woll + 0.01 * np.random.randint(1, 101) * (room_y - 2 * distance_from_woll)
        s_noise = np.array([x_noise, y_noise, high])
        d_noise = np.linalg.norm(s_noise - middle)
    
    # %% plot all
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, np.ones(180), label='Speaker circle')
    ax.plot(circ_mics_x, circ_mics_y, np.ones(180), label='Mic circle')
    ax.plot(line_x, line_y, np.ones(len(line_x)), label='Reference line')
    
    # Plot microphone positions (MATLAB indices 1,25,55,80,120,150,180 -> Python 0,24,54,79,119,149,179)
    mic_indices = [0, 24, 54, 79, 119, 149, 179]
    ax.scatter(
        [circ_mics_x[idx] for idx in mic_indices],
        [circ_mics_y[idx] for idx in mic_indices],
        [1] * 7,
        marker='o', s=50, label='Microphones'
    )
    
    t_noise_location = np.linspace(0, 2 * np.pi, 100)
    
    for i in range(num_jumps):
        x_noise_location = noise_R * np.sin(t_noise_location) + x2_temp[i]
        y_noise_location = noise_R * np.cos(t_noise_location) + y2_temp[i]
        z_noise_location = np.ones_like(t_noise_location)
        ax.plot(x_noise_location, y_noise_location, z_noise_location, 'g-', alpha=0.5)
        
        t_noise_location = np.linspace(0, 2 * np.pi, 100)
        x_noise_location = noise_R * np.sin(t_noise_location) + x1_temp[i]
        y_noise_location = noise_R * np.cos(t_noise_location) + y1_temp[i]
        z_noise_location = np.ones_like(t_noise_location)
        ax.plot(x_noise_location, y_noise_location, z_noise_location, 'b-', alpha=0.5)
        
        ax.scatter([x1[i], x2[i], s_noise[0]], [y1[i], y2[i], s_noise[1]], [high, high, high], marker='o', s=30)
    
    plotcube(L, [0, 0, 0], 0, [1, 1, 1], ax)  # use function plotcube
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Room with Speakers and Microphone Array')
    ax.legend(loc='upper left')
    
    py_path = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(py_path, 'Results')
    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, 'test_locations.png'), dpi=150)
    
    return s_first, label_first, s_second, label_second, s_noise, r
