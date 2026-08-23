"""Visual surface inspection (camera CV track).

See ADR-004. All heavy imports (cv2, ultralytics, albumentations) are lazy:
importing this package must not pull any of them.
"""
