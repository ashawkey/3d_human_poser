import os
import cv2
import math
import json
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation

def dot(x, y):
    return np.sum(x * y, -1, keepdims=True)
    
def length(x, eps=1e-20):
    return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))

def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

class Skeleton:

    def __init__(self):

        # init pose [18, 3], in [-1, 1]^3
        self.points3D = np.array([
            [-0.00313026,  0.16587697,  0.05414092],
            [-0.00857283,  0.1093518 , -0.00522604],
            [-0.06817748,  0.10397182, -0.00657925],
            [-0.11421658,  0.04033477,  0.00040599],
            [-0.15643744, -0.02915882,  0.03309248],
            [ 0.05288884,  0.10729481, -0.00067854],
            [ 0.10355149,  0.04464601, -0.00735265],
            [ 0.15390812, -0.02282556,  0.03085238],
            [ 0.03897187, -0.0403506 ,  0.00220192],
            [ 0.04027461, -0.15746351, -0.00187036],
            [ 0.04605377, -0.26837209, -0.0018945 ],
            [-0.0507806 , -0.04887162,  0.0022531 ],
            [-0.04873568, -0.16551849, -0.00128197],
            [-0.04840493, -0.27510208, -0.00128831],
            [-0.03098677,  0.19395538,  0.01987491],
            [ 0.01657042,  0.19560097,  0.02724142],
            [-0.05411603,  0.17336673, -0.01328044],
            [ 0.03733583,  0.16922003, -0.00946565]
        ], dtype=np.float32)

        self.name = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]

        # homogeneous
        self.points3D = np.concatenate([self.points3D, np.ones_like(self.points3D[:, :1])], axis=1) # [18, 4]

        # lines [17, 2]
        self.lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]], dtype=np.int32)

        # keypoint color [18, 3]
        self.colors = [[0, 0, 255], [255, 0, 0], [255, 170, 0], [255, 255, 0], [255, 85, 0], [170, 255, 0], 
                       [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], 
                       [0, 85, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    @property
    def center(self):
        return self.points3D[:, :3].mean(0)
    
    @property
    def center_upper(self):
        return self.points3D[0, :3]

    @property
    def torso_bbox(self):
        # valid_points = self.points3D[[0, 1, 8, 11], :3]
        valid_points = self.points3D[:, :3]
        # assure 3D thickness
        min_point = valid_points.min(0) - 0.1
        max_point = valid_points.max(0) + 0.1
        remedy_thickness = np.maximum(0, 0.8 - (max_point - min_point)) / 2
        min_point -= remedy_thickness
        max_point += remedy_thickness
        return min_point, max_point
    
    def write_json(self, path):

        with open(path, 'w') as f:
            d = {}
            for i in range(18):
                d[self.name[i]] = self.points3D[i, :3].tolist()
            json.dump(d, f)
    
    def load_json(self, path):

        with open(path, 'r') as f:
            d = json.load(f)
            for i in range(18):
                self.points3D[i, :3] = np.array(d[self.name[i]])

    def scale(self, delta):
        self.points3D[:, :3] *= 1.1 ** (-delta)

    def pan(self, rot, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.points3D[:, :3] += 0.0005 * rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])

    def draw(self, mvp, H, W, enable_occlusion=False):
        # mvp: [4, 4]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        points = self.points3D @ mvp.T # [18, 4]
        points = points[:, :3] / points[:, 3:] # NDC in [-1, 1]

        xs = (points[:, 0] + 1) / 2 * H # [18]
        ys = (points[:, 1] + 1) / 2 * W # [18]
        mask = (xs >= 0) & (xs < H) & (ys >= 0) & (ys < W)

        # hide certain keypoints based on empirical occlusion
        if enable_occlusion:
            # if nose is further than both eyes, it's back face
            if points[0, 2] > points[-3, 2] and points[0, 2] > points[-4, 2]:
                mask[0] = False
                mask[-3] = False
                mask[-4] = False
            # if left ear is in the left of neck and right of right ear, hide it... and so on
            if xs[-2] > xs[0] and xs[-2] < xs[-1]:
                mask[-2] = False
            if xs[-4] > xs[-3] and xs[-4] < xs[-1]:
                mask[-4] = False
            if xs[-1] > xs[-2] and xs[-1] < xs[0]:
                mask[-1] = False
            if xs[-3] > xs[-2] and xs[-3] < xs[-4]:
                mask[-3] = False

        # 18 points
        for i in range(18):
            if not mask[i]: continue
            cv2.circle(canvas, (int(xs[i]), int(ys[i])), 4, self.colors[i], thickness=-1)

        # 17 lines
        for i in range(17):
            cur_canvas = canvas.copy()
            if not mask[self.lines[i]].all(): 
                continue
            X = xs[self.lines[i]]
            Y = ys[self.lines[i]]
            mY = np.mean(Y)
            mX = np.mean(X)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1)
            
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
            
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        canvas = canvas.astype(np.float32) / 255
        return canvas, np.stack([xs, ys], axis=1)
        

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = Rotation.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (1.414 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)
    
    # projection (perspective)
    @property
    def perspective(self):
        y = 1.414 / 2 * np.tan(np.radians(self.fovy) / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )
    
    def from_angle(self, elevation, azimuth, is_degree=True):
        # elevation: [-90, 90], from +y --> -y
        # azimuth: [0, 360], from +z --> -x --> -z --> +x --> +z
        if is_degree:
            elevation = np.deg2rad(elevation)
            azimuth = np.deg2rad(azimuth)
        x = self.radius * np.cos(elevation) * np.sin(azimuth)
        y = self.radius * np.sin(elevation)
        z = self.radius * np.cos(elevation) * np.cos(azimuth)
        campos = np.array([x, y, z])  # [N, 3] 
        forward_vector = safe_normalize(campos)
        up_vector = self.up
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
        rot_mat = np.stack([right_vector, up_vector, forward_vector], axis=1)
        self.rot = Rotation.from_matrix(rot_mat)


    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = Rotation.from_rotvec(rotvec_x) * Rotation.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])


class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.skel = Skeleton()
        
        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation

        self.save_image_path = 'pose.png'
        self.save_json_path = 'pose.json'
        self.mouse_loc = np.array([0, 0])
        self.points2D = None # [18, 2]
        self.point_idx = 0
        self.drag_sensitivity = 0.0001
        self.pan_scale_skel = True
        self.enable_occlusion = True
        
        dpg.create_context()
        self.register_dpg()
        self.step()
        

    def __del__(self):
        dpg.destroy_context()


    def step(self):

        if self.need_update:
        
            # mvp
            mv = self.cam.view # [4, 4]
            proj = self.cam.perspective # [4, 4]
            mvp = proj @ mv

            # render our openpose image, somehow
            self.render_buffer, self.points2D = self.skel.draw(mvp, self.H, self.W, enable_occlusion=self.enable_occlusion)
        
            self.need_update = False
            
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(label="Viewer", tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=-1, height=-1):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # save image    
            def callback_save_image(sender, app_data):
                image = (self.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.save_image_path, image)
                print(f'[INFO] write image to {self.save_image_path}')
            
            def callback_set_save_image_path(sender, app_data):
                self.save_image_path = app_data
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="save image", tag="_button_save_image", callback=callback_save_image)
                dpg.bind_item_theme("_button_save_image", theme_button)

                dpg.add_input_text(label="", default_value=self.save_image_path, callback=callback_set_save_image_path)
            
            # save json
            def callback_save_json(sender, app_data):
                self.skel.write_json(self.save_json_path)
                print(f'[INFO] write json to {self.save_json_path}')
            
            def callback_set_save_json_path(sender, app_data):
                self.save_json_path = app_data
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="save json", tag="_button_save_json", callback=callback_save_json)
                dpg.bind_item_theme("_button_save_json", theme_button)

                dpg.add_input_text(label="", default_value=self.save_json_path, callback=callback_set_save_json_path)

            # pan/scale mode
            def callback_set_pan_scale_mode(sender, app_data):
                self.pan_scale_skel = not self.pan_scale_skel

            dpg.add_checkbox(label="pan/scale skeleton", default_value=self.pan_scale_skel, callback=callback_set_pan_scale_mode)

            # backview mode
            def callback_set_occlusion_mode(sender, app_data):
                self.enable_occlusion = not self.enable_occlusion
                self.need_update = True

            dpg.add_checkbox(label="use occlusion", default_value=self.enable_occlusion, callback=callback_set_occlusion_mode)

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True

            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

            # drag sensitivity
            def callback_set_drag_sensitivity(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True

            dpg.add_slider_float(label="drag sensitivity", min_value=0.000001, max_value=0.001, format="%f", default_value=self.drag_sensitivity, callback=callback_set_drag_sensitivity)

              
        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data
            
            if self.pan_scale_skel:
                self.skel.scale(delta)
            else:
                self.cam.scale(delta)

            self.need_update = True


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            if self.pan_scale_skel:
                self.skel.pan(self.cam.rot, dx, dy)
            else:
                self.cam.pan(dx, dy)

            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        def callback_skel_select(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return
            
            # determine the selected keypoint from mouse_loc
            if self.points2D is None: return # not prepared

            dist = np.linalg.norm(self.points2D - self.mouse_loc, axis=1) # [18]
            self.point_idx = np.argmin(dist)

        
        def callback_skel_drag(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            # 2D to 3D delta
            dx = app_data[1]
            dy = app_data[2]
        
            self.skel.points3D[self.point_idx, :3] += self.drag_sensitivity * self.cam.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, 0])

            self.need_update = True


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

            # for skeleton editing
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=callback_skel_select)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_skel_drag)

        
        dpg.create_viewport(title='pose viewer', resizable=False, width=self.W, height=self.H)
        
        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)
        dpg.focus_item("_primary_window")

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):

        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--W', type=int, default=512, help="GUI width")
    parser.add_argument('--H', type=int, default=512, help="GUI height")
    parser.add_argument('--load', type=str, default=None, help="path to load a json pose")
    parser.add_argument('--save', type=str, default=None, help="path to render and save pose images")
    parser.add_argument('--radius', type=float, default=2.7, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=18.8, help="default GUI camera fovy")

    opt = parser.parse_args()

    gui = GUI(opt)

    if opt.load is not None:
        print(f'[INFO] load from {opt.load}')
        gui.skel.load_json(opt.load)
        gui.need_update = True
    
    if opt.save is not None:
        os.makedirs(opt.save, exist_ok=True)
        # render from fixed views and save all images
        elevation = [-10, 0, 10, 20, 30]
        azimuth = np.arange(0, 360, dtype=np.int32)
        for ele in tqdm.tqdm(elevation):
            for azi in tqdm.tqdm(azimuth):
                gui.cam.from_angle(ele, azi)
                gui.need_update = True
                gui.step()
                dpg.render_dearpygui_frame()
                image = (gui.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(opt.save, f'{ele}_{azi:04d}.jpg'), image)
    else:
        gui.render()
