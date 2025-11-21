from scipy.ndimage import distance_transform_edt
import numba as nb
import numpy as np
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

def check_inside(x0, y0, x1, y1, nx, ny):
    # Completely outside if the box is entirely to the left, right, above, or below.
    if x1 <= 0 or x0 >= nx or y1 <= 0 or y0 >= ny:
        dx = 0
        dy = 0
        if x1 <= 0:
            dx = abs(x1)
        elif x0 >= nx:
            dx = x0 - nx
        if y1 <= 0:
            dy = abs(y1)
        elif y0 >= ny:
            dy = y0 - ny
        print("Box is completely outside the boundaries.")
        return False
    else:
        return True

@nb.njit
def gaussian_border_weights(Result, Input, distance, x_min, y_min, rx, ry, ramp_width):
    nx, ny = Result.shape  # get dimensions of the array

    sigma = ramp_width / 3.0

    # Loop over every pixel in the global array.
    for i in range(-ramp_width,rx+ramp_width):
        for j in range(-ramp_width,ry+ramp_width):
            x_grid = x_min + i 
            y_grid = y_min + j

            if x_grid >= 0 and x_grid <= nx-1 and y_grid >= 0 and y_grid <= ny-1:
                Result[x_grid, y_grid] = Input[x_grid, y_grid] * np.exp(-0.5 * (distance[x_grid, y_grid] / sigma)**2)
            
    return Result

def insert_region_into_global(output, data, region_data, x_mesh, y_mesh, ramp_width):
    nx, ny = output.shape

    # Determine the bounding box from the meshgrid arrays.
    rx, ry = region_data.shape
    
    x_min = int(x_mesh.min())
    x_max = x_min + rx
    y_min = int(y_mesh.min())
    y_max = y_min + ry

    isInside = check_inside(x_min, y_min, x_max, y_max, nx, ny)

    if isInside:

        # Compute the intersection with the global domain:
        global_x_min = max(x_min, 0)
        global_x_max = min(x_max, nx)
        global_y_min = max(y_min, 0)
        global_y_max = min(y_max, ny)

        # Now, determine the corresponding slice in region_data.
        # For example, if x_min < 0, then the starting index in region_data is offset by -x_min.
        region_x_start = global_x_min - x_min
        region_x_end   = region_x_start + (global_x_max - global_x_min)
        region_y_start = global_y_min - y_min
        region_y_end   = region_y_start + (global_y_max - global_y_min)

        # Now assign:
        output[global_x_min:global_x_max, global_y_min:global_y_max] = region_data[region_x_start:region_x_end, region_y_start:region_y_end]
        mask = output == 0
        distances = distance_transform_edt(mask)
        output = gaussian_border_weights(output, data, distances, x_min, y_min, rx, ry, ramp_width)

    return output

# ---------------------------
# Create functions for control events.

class AcceptState:
    def __init__(self, initial=True):
        self.state = initial

    def toggle(self):
        self.state = not self.state
        return self.state

def make_toggle_accept(acceptBtn, state_obj):
    def toggle_accept():
        new_state = state_obj.toggle()
        if new_state:
            acceptBtn.setText("Accept")
        else:
            acceptBtn.setText("Reject")
    return toggle_accept

def update(w_data, seismogram, imv1, imv2, imv3, roi, perc, percSeis, accept_state, ramp_width):

    # global w_data, seismogram, imv1, imv2, roi, perc, percSeis, accept_state

    # Extract ROI region and mapped coordinates from w_data.
    data, coords = roi.getArrayRegion(w_data, imv1.imageItem, axes=(0, 1), returnMappedCoords=True)

    output = np.zeros_like(w_data, dtype=complex)

    # Insert region_data into the output using your insertion routine.
    inserted = insert_region_into_global(output, w_data, data, coords[0], coords[1], ramp_width)
    
    # # Depending on the accept/reject mode, set Result accordingly.
    if accept_state:
        # Accept: use the inserted ROI.
        Result = inserted
    else:
        # Reject: subtract the inserted ROI from the original w_data.
        Result = w_data - inserted

    # Optionally, process Result further (for example, perform an inverse FFT).
    processed = np.real(np.fft.irfft2(np.fft.fftshift(Result.T, axes=1),
                                       axes=(1, 0), s=seismogram.T.shape))

    imv2.setImage(np.abs(Result))
    imv2.setLevels(0, perc)
    imv2.setHistogramRange(0, perc)
    imv2.getView().invertY(False)

    imv3.setImage(processed.T)
    imv3.setHistogramRange(-percSeis, percSeis)
    imv3.setLevels(-percSeis, percSeis)

def init_fk_window():
    win = QtWidgets.QMainWindow()
    win.resize(1500, 1200)
    win.setWindowTitle('Polygon ROI Data Extraction')
    cw = QtWidgets.QWidget()
    win.setCentralWidget(cw)
    
    return win, cw

def init_control_widget():
    controlWidget = QtWidgets.QWidget()
    controlLayout = QtWidgets.QHBoxLayout()
    controlWidget.setLayout(controlLayout)

    return controlWidget, controlLayout

def add_Buttons_to_control_layout(controlLayout, ramp_width = 40):
    rampSpinBox = QtWidgets.QSpinBox()
    rampSpinBox.setRange(1, 200)  # you can adjust the range as needed
    rampSpinBox.setValue(ramp_width)
    rampSpinBox.setSuffix(" px")
    controlLayout.addWidget(rampSpinBox)

    # Toggle button for accept/reject.
    acceptBtn = QtWidgets.QPushButton("Accept")
    controlLayout.addWidget(acceptBtn)
    

    # A button to save the current output variable.
    saveBtn = QtWidgets.QPushButton("Save Variable")
    controlLayout.addWidget(saveBtn)

    return controlLayout, acceptBtn, saveBtn

def create_plot_items():
    plotItem = pg.PlotItem()
    plotItem2 = pg.PlotItem()
    plotItem3 = pg.PlotItem()

    plotItem.setLabel('left', 'Frequency (Hz)')
    plotItem.setLabel('bottom', 'Wavenumber (1/m)')

    plotItem2.setLabel('left', 'Frequency (Hz)')
    plotItem2.setLabel('bottom', 'Wavenumber (1/m)')

    plotItem3.setLabel('left', 'Time (s)')
    plotItem3.setLabel('bottom', 'Channel')

    return plotItem, plotItem2, plotItem3

    # y_ticks = [(0, f'{0:.2f}'),
    #         ((nt)/2, f'{(nt*dt)/2:.2f}'),
    #         (nt, f'{nt*dt:.2f}')]

    # plotItem3.getAxis('left').setTicks([y_ticks])

    # x_min = kx[0]
    # x_max = kx[-1]

    # x_ticks = [(0, f'{x_min:.2f}'),
    #         ((len(kx))/2, f'{(x_min+x_max)/2:.2f}'),
    #         (len(kx), f'{x_max:.2f}')]

    # y_min = freq[0]
    # y_max = freq[-1]

    # y_ticks = [(0, f'{y_min:.2f}'),
    #         ((len(freq))/2, f'{(y_min+y_max)/2:.2f}'),
    #         (len(freq), f'{y_max:.2f}')]

    # plotItem.getAxis('left').setTicks([y_ticks])
    # plotItem.getAxis('bottom').setTicks([x_ticks])
    # plotItem2.getAxis('left').setTicks([y_ticks])
    # plotItem2.getAxis('bottom').setTicks([x_ticks])

def get_color_map(map='jet', source='matplotlib'):
    return pg.colormap.get(map, source=source, skipCache=True)

def set_images_data(imv1, imv2, imv3, data_abs, seismogram, perc, percSeis):

    # ---------------------------
    # Display the Data in images
    # ---------------------------
    imv1.setImage(data_abs)
    imv1.setHistogramRange(0, perc)
    imv1.setLevels(0, perc)
    imv1.getView().invertY(False)
    imv1.getView().invertX(False)
    imv1.ui.roiBtn.hide()
    imv1.ui.menuBtn.hide()

    imv2.setLevels(0, perc)
    imv2.getView().invertY(False)
    imv2.getView().invertX(False)
    imv2.ui.roiBtn.hide()
    imv2.ui.menuBtn.hide()

    imv3.setImage(seismogram.T)
    imv3.setLevels(-percSeis, percSeis)
    imv3.getView().invertY(True)
    imv3.ui.roiBtn.hide()
    imv3.ui.menuBtn.hide()



class GraphPolyLine(pg.PolyLineROI):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def checkPointMove(self, handle, pos, modifiers):
        if self.maxBounds is not None:
            pt = self.getViewBox().mapSceneToView(pos)
        
            if not self.maxBounds.contains(pt.x(), pt.y()):
                return False

        return True

def create_ROI():
    roi_width = 400
    roi_height = 400

    roi = GraphPolyLine([[0, 0],
                      [0 + roi_width, 0],
                      [0 + roi_width, 0 + roi_height],
                      [0, 0 + roi_height]],
                     closed=True, movable=True) #, maxBounds = QtCore.QRectF(0, 0, nx, ny)
    
    return roi

# def run_FK_window(data, geometry, dt, key_geometry='FFID', id_list = [1000]):
#     print("entrei")

#     app = pg.mkQApp("FK Filter")

#     win, cw = init_fk_window()
#     layout = QtWidgets.QGridLayout()

#     controlWidget, controlLayout = init_control_widget()
    
#     controlLayout, acceptBtn, saveBtn = add_Buttons_to_control_layout(controlLayout)
    
#     # Create an instance to hold the accept/reject state:
#     accept_state_obj = AcceptState(True)

#     # Connect the button using the closure:
#     acceptBtn.clicked.connect(make_toggle_accept(acceptBtn, accept_state_obj))

#     layout.addWidget(controlWidget, 1, 0, 1, 3)

#     cw.setLayout(layout)

#     plot1, plot2, plot3 = create_plot_items()

#     imv1 = pg.ImageView(view=plot1)
#     imv2 = pg.ImageView(view=plot2)
#     imv3 = pg.ImageView(view=plot3)

#     imv1.setColorMap(get_color_map('jet','matplotlib'))
#     imv2.setColorMap(get_color_map('jet','matplotlib'))
#     imv3.setColorMap(get_color_map('gray_r','matplotlib'))

#     layout.addWidget(imv1, 0, 0)
#     layout.addWidget(imv2, 0, 1)
#     layout.addWidget(imv3, 0, 2)
#     win.show()

    
#     roi = create_ROI()
#     imv1.addItem(roi)

    

#     for id in id_list:

#         df = geometry[geometry[key_geometry] == id]
#         seismogram = data[:,df.index.to_numpy()]

#         nt,nr = seismogram.shape

#         w_data = np.fft.rfft2(seismogram,axes=(1,0))
#         kx = np.fft.fftshift(np.fft.fftfreq(nr,d=12.5))
#         freq = np.fft.rfftfreq(nt,d=dt)

#         perc=np.percentile(np.abs(w_data),95)
#         percSeis=np.percentile(seismogram,96)

#         w_data = np.fft.fftshift(w_data,axes=1).T
#         data_abs=np.abs(w_data)

#         set_images_data(imv1, imv2, imv3, data_abs, seismogram, perc, percSeis)

#         imv1.imageItem.setRect(QtCore.QRectF(0, 0, w_data.shape[0], w_data.shape[1]))

#         if isinstance(seismogram, type(None)):
#             print("entrei no none")

#         timer = pg.QtCore.QTimer()
#         timer.timeout.connect(lambda: update(w_data, seismogram, imv1, imv2, imv3, roi, perc, percSeis, 
#                                              accept_state_obj.state, 
#                                              controlLayout.itemAt(0).widget().value()))
#         timer.start(500)
#         pg.exec()
#         timer.stop()

#     # timer.stop()
#     win.deleteLater()
#     win.close()

global current_id_index
def run_FK_window(data, geometry, dt, key_geometry='FFID', id_list=[300, 800,1348]):
    print("Starting FK Window")
    global current_id_index
    current_id_index = 0
    app = pg.mkQApp("FK Filter")
    win, cw = init_fk_window()
    layout = QtWidgets.QGridLayout()
    cw.setLayout(layout)
    
    controlWidget, controlLayout = init_control_widget()
    controlLayout, acceptBtn, saveBtn = add_Buttons_to_control_layout(controlLayout, ramp_width=40)
    layout.addWidget(controlWidget, 1, 0, 1, 3)
    
    # Create an instance to hold the accept/reject state.
    accept_state_obj = AcceptState(True)
    acceptBtn.clicked.connect(make_toggle_accept(acceptBtn, accept_state_obj))
    
    plot1, plot2, plot3 = create_plot_items()
    imv1 = pg.ImageView(view=plot1)
    imv2 = pg.ImageView(view=plot2)
    imv3 = pg.ImageView(view=plot3)
    
    imv1.setColorMap(get_color_map('jet','matplotlib'))
    imv2.setColorMap(get_color_map('jet','matplotlib'))
    imv3.setColorMap(get_color_map('gray_r','matplotlib'))
    
    layout.addWidget(imv1, 0, 0)
    layout.addWidget(imv2, 0, 1)
    layout.addWidget(imv3, 0, 2)
    win.show()
    
    roi = create_ROI()
    imv1.addItem(roi)
    
    # Global index for id_list processing.
    
    num_ids = len(id_list)
    
    # Function to process current shot
    def process_shot():
        global seismogram, w_data, data_abs, perc, percSeis
        current_id = id_list[current_id_index]
        print("Processing id:", current_id)
        df = geometry[geometry[key_geometry] == current_id]
        seismogram = data[:, df.index.to_numpy()]
        nt, nr = seismogram.shape
        w_data = np.fft.rfft2(seismogram, axes=(1, 0))
        kx = np.fft.fftshift(np.fft.fftfreq(nr, d=12.5))
        freq = np.fft.rfftfreq(nt, d=dt)
        perc = np.percentile(np.abs(w_data), 95)
        percSeis = np.percentile(seismogram, 96)
        w_data = np.fft.fftshift(w_data, axes=1).T
        data_abs = np.abs(w_data)
        set_images_data(imv1, imv2, imv3, data_abs, seismogram, perc, percSeis)
        imv1.imageItem.setRect(QtCore.QRectF(0, 0, w_data.shape[0], w_data.shape[1]))
    
    process_shot()  # Process first shot

    # Timer to call update() every 500 ms.
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: update(w_data, seismogram, imv1, imv2, imv3, roi, perc, percSeis, 
                                          accept_state_obj.state, controlLayout.itemAt(0).widget().value()))
    timer.start(500)

    # Save Variable button: when clicked, save the current variable and then move to next shot.
    def save_and_next():
        global current_id_index
        # For example, save the current distances variable (or any variable you wish).
        # Here, we save the current output from imv2's imageItem.
        # current_output = imv2.imageItem.image()
        # np.save("output_shot_{}.npy".format(id_list[current_id_index]), current_output)
        # print("Saved shot id:", id_list[current_id_index])
        # Advance to next shot.
        current_id_index += 1
        if current_id_index < len(id_list):
            process_shot()
        else:
            print("All shots processed.")
            win.deleteLater()
            win.close()
    
    saveBtn.clicked.connect(save_and_next)
    
    pg.exec()
    timer.stop()
    