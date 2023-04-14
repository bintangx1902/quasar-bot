def postprocess_outputs(predictions, np):
    box_x, box_y, box_w, box_h, class_index = None, None, None, None, np.argmax(predictions)
    return box_x, box_y, box_w, box_h, class_index


def draw_box(frame, box_x, box_y, box_w, box_h, class_label, cv2):
    f_copy = frame.copy()
    if box_x is not None and box_y is not None and box_w is not None and box_h is not None:
        cv2.rectangle(f_copy, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
        if class_label is not None:
            cv2.putText(f_copy, class_label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return f_copy


