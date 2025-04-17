import cv2
import torch
import numpy as np
from ultralytics import solutions
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
import supervision as sv
from shapely.geometry import Point, Polygon

# Pass region as dictionary

region_colors = [
    (255, 0, 255),
    (0, 255, 255),
    (86, 0, 254),
    (0, 128, 255),
    (235, 183, 0),
    (255, 34, 134)
]
vehicle_color = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 128)
]

class MultipleObjectCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ktra xem trong dictionary self.CFG co key "regions" -> lay self.CFG["regions"]
        # Neu khong thi lay region khong co region thi tra ve rong.
        cfg_regions = self.CFG["regions"] if "regions" in self.CFG else self.CFG.get("region", [])
        # Gan ket qua vao self.regions
        self.regions = cfg_regions

        # Prepare separate counters and sets for each region.
        # Bo dem va tap hop cho moi region

        self.in_counts = [0] * len(self.regions)    # Dem trong moi khu vuc co bao nhieu xe di vao
        self.out_counts = [0] * len(self.regions)   # Dem xe di ra

        # Tao set id cua cac doi tuong da duoc dem trong moi vung.
        self.counted_ids = [set() for _ in range(len(self.regions))]    # Set id cho xe da dem tranh trung lap
        # Danh dau cac vung
        self.region_initialized = False
        
        # Hien thi ket qua 
        self.show_in = self.CFG.get("show_in")
        self.show_out = self.CFG.get("show_out")
    
    # Khoi tao vung
    def initialize_region_geometry(self):
        self.Point = Point
        self.Polygon = Polygon

    def count_objects_in_region(self, region_idx, region_points, current_centroid, track_id, prev_position, cls):
        # Ktra vi tri object co trong vung Polygon khong
        if prev_position is None or track_id in self.counted_ids[region_idx]:
            return
        polygon = self.Polygon(region_points)

        # Ktra toa do cua diem trung tam cua doi tuong xem co nam trong da giac cua lan duong do hay khong
        if polygon.contains(self.Point(current_centroid)):
            xs = [pt[0] for pt in region_points]
            ys = [pt[1] for pt in region_points]

            region_width = max(xs) - min(xs)
            region_height = max(ys) - min(ys)

            # Ktra xem dang di vao hay di ra
            going_in = False
            if region_width < region_height and current_centroid[0] > prev_position[0]:
                going_in = True
            elif region_width >= region_height and current_centroid[1] > prev_position[1]:
                going_in = True
            
            if going_in:
                self.in_counts[region_idx] += 1
            else:
                self.out_counts[region_idx] += 1
            
            self.counted_ids[region_idx].add(track_id)

    # Hien thi tong so doi tuong trong moi vung.
    def display_counts(self, plot_im):
        # Display region-specific out count in the center of each region.
        for i, region_points in enumerate(self.regions):
            xs = [pt[0] for pt in region_points]
            ys = [pt[1] for pt in region_points]
            cx = int(sum(xs)/len(xs))
            cy = int(sum(ys)/len(ys))

            text_str = f"{self.in_counts[i] + self.out_counts[i]}"
            cv2.putText(
                plot_im,
                text_str,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                5
            )

    
    def process(self, frame):
        """
        Xem co xe nao moi di vao, neu co tang count_id +1
        """
        if not self.region_initialized:
            self.initialize_region_geometry()
            self.region_initialized = True

        # Lay bounding box  
        self.extract_tracks(frame)

        # Initilize
        self.annotator = SolutionAnnotator(frame, line_width=self.line_width)

        for idx, region_points in enumerate(self.regions):
            color = region_colors[idx]

            # Ve hinh da giac xung quanh
            self.annotator.draw_region(
                reg_pts=region_points,
                color=color,
                thickness=self.line_width * 2
            )

            b, g, r = color
            # ve pixel ben trong moi da giac
            frame = sv.draw_filled_polygon(
                scene=frame,
                polygon=np.array(region_points),
                color=sv.Color(r=r, g=g, b=b),
                opacity=0.25
            )

        # Duyet qua lan luot cac doi tuong trong cac vung, ve bounding box quanh doi tuong
        # track_id de theo doi tung doi tuong
        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=self.names[cls], color=vehicle_color[object_classes[0]])                
            self.store_tracking_history(track_id, box)

            # tinh toa do trung tam cua tung doi tuong
            current_centroid = (
                (box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2
            )

            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]

                #Ktra xem cac doi tuong co nam trong cac khu vuc duoc dinh nghia ben tren
            for r_idx, region_points in enumerate(self.regions):
                self.count_objects_in_region(
                    region_idx=r_idx,
                    region_points=region_points,
                    current_centroid=current_centroid,
                    track_id=track_id,
                    prev_position=prev_position,
                    cls=cls
                )

        plot_im = self.annotator.result()

        self.display_counts(plot_im)

        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            total_tracks=len(self.track_ids),
        )
        

if __name__ == '__main__':
    object_classes = [2, 5, 7]
    video_path = ("data/Video.mp4")
    cap = cv2.VideoCapture(video_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    region_points = [
        [[13, 716], [0, 705], [2, 614], [469, 348], [505, 347]],
        [[504, 347], [561, 347], [252, 716], [16, 716]],
        [[562, 348], [608, 349], [485, 716], [246, 716]],
        [[678, 353], [728, 354], [974, 717], [765, 715]],
        [[728, 350], [779, 353], [1190, 715], [973, 715]],
        [[780, 352], [825, 355], [1275, 680], [1272, 713], [1189, 713]]
    ]

    counter = MultipleObjectCounter(
        show = True, #display the frame
        region=region_points,    #pass region points
        model="yolo11n.pt",
        classes = object_classes,
    )
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Out Loop.")
            break
        results = counter.process(frame)
        frame = results.plot_im
        video_writer.write(results.plot_im)
    
cap.release()
video_writer.release()
cv2.destroyAllWindows()