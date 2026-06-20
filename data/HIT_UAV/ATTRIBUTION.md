# HIT-UAV — Attribution and License

This `HIT_UAV/` folder redistributes the **HIT-UAV** high-altitude infrared
thermal dataset. It is included under the terms of its license; this file
provides the attribution those terms require.

## License
**Creative Commons Attribution 4.0 International (CC BY 4.0)** —
<https://creativecommons.org/licenses/by/4.0/>

Under CC BY 4.0 you are free to share and adapt the material, including
commercially, provided you give appropriate credit, link to the license, and
indicate if changes were made.

## Original creators (please cite)
Suo, J., Wang, T., Zhang, X., Chen, H., Zhou, W., and Shi, W.
*HIT-UAV: A high-altitude infrared thermal dataset for unmanned aerial
vehicle-based object detection.* Scientific Data **10**, 227 (2023).
DOI: <https://doi.org/10.1038/s41597-023-02066-6>

Original dataset repository:
<https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset>

```bibtex
@article{hituav,
  author  = {Suo, Jiang and Wang, Tianbao and Zhang, Xiao and Chen, Haiyou and Zhou, Wei and Shi, Wugen},
  title   = {{HIT-UAV}: A high-altitude infrared thermal dataset for unmanned aerial vehicle-based object detection},
  journal = {Scientific Data},
  volume  = {10},
  pages   = {227},
  year    = {2023},
  doi     = {10.1038/s41597-023-02066-6}
}
```

## Source of this copy
Obtained from Roboflow Universe (a third-party re-export):
<https://universe.roboflow.com/hituav-dataset-for-yolov5/hit-uav-for-yolov5-rjdrs>
(see `README.roboflow.txt` and `README.dataset.txt`, retained in this folder).

## Changes made to the original
This copy is **not** the pristine original; the following modifications were
applied and are disclosed as required by CC BY 4.0:
- Re-exported via Roboflow with EXIF-orientation stripping and auto-orientation;
- Images resized to 640×640 (stretch); no augmentation applied;
- Annotations converted to YOLO format;
- Dataset/class configuration adjusted for this project (`data.yaml` /
  `data_fixed.yaml`).

No claim of ownership is made over the HIT-UAV dataset. All rights remain with
the original authors under CC BY 4.0.
