# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH


class STrack(BaseTrack):
    """
    单目标跟踪表示，使用卡尔曼滤波进行状态估计。
    
    该类负责存储关于单个跟踪目标的所有信息，并基于卡尔曼滤波执行状态更新和预测。
    
    属性:
        shared_kalman (KalmanFilterXYAH): 所有STrack实例共享的卡尔曼滤波器，用于预测。
        _tlwh (np.ndarray): 存储边界框左上角坐标及宽高的私有属性。
        kalman_filter (KalmanFilterXYAH): 用于此特定目标跟踪的卡尔曼滤波器实例。
        mean (np.ndarray): 状态估计均值向量。
        covariance (np.ndarray): 状态估计协方差。
        is_activated (bool): 表示跟踪是否已激活的布尔标志。
        score (float): 跟踪的置信度分数。
        tracklet_len (int): 跟踪序列的长度。
        cls (Any): 目标的类别标签。
        idx (int): 目标的索引或标识符。
        frame_id (int): 当前帧ID。
        start_frame (int): 目标首次被检测到的帧。
    """

    shared_kalman = KalmanFilterXYAH()  # 共享的卡尔曼滤波器

    def __init__(self, xywh, score, cls):
        """
        初始化一个新的STrack实例。
        
        参数:
            xywh (List[float]): 边界框坐标和尺寸，格式为(x, y, w, h, [a], idx)
                其中(x, y)是中心点，(w, h)是宽高，[a]是可选的宽高比，idx是ID
            score (float): 检测的置信度分数
            cls (Any): 检测到的目标的类别标签
        """
        super().__init__()
        # xywh+idx 或 xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """使用卡尔曼滤波器预测目标的下一个状态（均值和协方差）。"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """对提供的STrack实例列表执行多目标预测跟踪，使用卡尔曼滤波器。"""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """使用单应性矩阵更新多个轨迹的状态位置和协方差。"""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """激活一个新的跟踪目标，使用提供的卡尔曼滤波器并初始化其状态和协方差。"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """重新激活之前丢失的跟踪目标，使用新的检测数据更新其状态和属性。"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track, frame_id):
        """
        更新匹配轨迹的状态。
        
        参数:
            new_track (STrack): 包含更新信息的新轨迹
            frame_id (int): 当前帧的ID
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        """将边界框从左上角-宽高格式转换为中心点-宽高比-高度格式。"""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        """从当前状态估计返回左上角-宽高格式的边界框。"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        """将边界框从(左上x, 左上y, 宽, 高)转换为(最小x, 最小y, 最大x, 最大y)格式。"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """将边界框从tlwh格式转换为中心点-宽高比-高度(xyah)格式。"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        """返回边界框的当前位置，格式为(中心x, 中心y, 宽, 高)。"""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        """返回位置，格式为(中心x, 中心y, 宽, 高, 角度)，如果角度缺失则发出警告。"""
        if self.angle is None:
            LOGGER.warning("WARNING ⚠️ `angle` 属性未找到，改为返回 `xywh`。")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        """返回当前跟踪结果，采用适当的边界框格式。"""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]

    def __repr__(self):
        """返回STrack对象的字符串表示，包括起始帧、结束帧和跟踪ID。"""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """
    BYTETracker: 基于YOLOv8的目标检测与跟踪算法。
    
    此类封装了初始化、更新和管理视频序列中检测到的目标轨迹的功能。
    它维护已跟踪、丢失和已移除轨迹的状态，使用卡尔曼滤波预测新的目标位置，并执行数据关联。
    
    属性:
        tracked_stracks (List[STrack]): 成功激活的轨迹列表。
        lost_stracks (List[STrack]): 丢失的轨迹列表。
        removed_stracks (List[STrack]): 已移除的轨迹列表。
        frame_id (int): 当前帧ID。
        args (Namespace): 命令行参数。
        max_time_lost (int): 一个轨迹被视为"丢失"的最大帧数。
        kalman_filter (KalmanFilterXYAH): 卡尔曼滤波器对象。
    """

    def __init__(self, args, frame_rate=30):
        """
        初始化BYTETracker实例进行目标跟踪。
        
        参数:
            args (Namespace): 包含跟踪参数的命令行参数。
            frame_rate (int): 视频序列的帧率。
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img=None):
        """用新的检测结果更新跟踪器并返回当前跟踪目标列表。"""
        self.frame_id += 1
        activated_stracks = []  # 当前帧激活的轨迹
        refind_stracks = []     # 当前帧重新找到的轨迹
        lost_stracks = []       # 当前帧丢失的轨迹
        removed_stracks = []    # 当前帧移除的轨迹

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        # 添加索引
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls

        # 高分检测框过滤
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        # 低分检测框过滤
        inds_second = inds_low & inds_high
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]

        # 初始化跟踪
        detections = self.init_track(dets, scores_keep, cls_keep, img)
        # 将新检测到的轨迹添加到tracked_stracks
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        # 步骤2: 第一次关联，使用高分检测框
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # 使用卡尔曼滤波预测当前位置
        self.multi_predict(strack_pool)
        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # 计算轨迹和检测之间的距离
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # 处理匹配的轨迹和检测
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # 步骤3: 第二次关联，使用低分检测框关联未跟踪到的轨迹
        detections_second = self.init_track(dets_second, scores_second, cls_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # TODO
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 处理未匹配的轨迹，标记为丢失
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # 处理未确认的轨迹，通常是只有一个开始帧的轨迹
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        # 步骤4: 初始化新轨迹
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        # 步骤5: 更新状态
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 最终更新跟踪器状态
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # 将移除的轨迹限制为最多1000个

        # 返回跟踪结果
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self):
        """返回用于跟踪边界框的卡尔曼滤波器对象。"""
        return KalmanFilterXYAH()

    def init_track(self, dets, scores, cls, img=None):
        """使用给定的检测、分数和类别标签初始化目标跟踪。"""
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []  # 检测结果

    def get_dists(self, tracks, detections):
        """计算轨迹和检测之间的距离，使用IoU并可选地融合分数。"""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        """使用卡尔曼滤波器预测多个轨迹的下一个状态。"""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """重置STrack实例的ID计数器，确保跟踪会话之间的唯一轨迹ID。"""
        STrack.reset_id()

    def reset(self):
        """重置跟踪器，清除所有已跟踪、丢失和已移除的轨迹，并重新初始化卡尔曼滤波器。"""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        """将两个STrack对象列表合并为一个列表，确保基于轨迹ID没有重复。"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        """从第一个列表中过滤出存在于第二个列表中的轨迹。"""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        """基于交并比(IoU)距离从两个列表中移除重复的轨迹。"""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb
