import os
import cv2
import numpy as np
import threading
import time
import json
import serial
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import pyttsx3
import speech_recognition as sr
from queue import Queue
import sqlite3
from datetime import datetime

# ===== 配置和常量 =====
@dataclass
class RobotConfig:
    """机器人配置类"""
    # 硬件配置
    SERIAL_PORT: str = '/dev/ttyUSB0'
    BAUD_RATE: int = 115200
    CAMERA_ID: int = 0
    
    # 性能指标
    CLEANING_COVERAGE_RATE: float = 0.9  # 清扫覆盖率 >90%
    OBJECT_RECOGNITION_ACCURACY: float = 0.9  # 物品识别准确率 ≥90%
    VOICE_ALERT_ERROR_RATE: float = 0.05  # 语音提醒误报率 <5%
    MR_RESPONSE_TIME: float = 0.05  # 混合现实交互响应时间 <50ms
    BATTERY_LIFE: int = 9  # 续航时长 >9小时
    
    # 路径规划参数
    MAP_RESOLUTION: float = 0.05  # 地图分辨率 5cm
    SAFE_DISTANCE: float = 0.3  # 安全距离 30cm
    CLEANING_WIDTH: float = 0.25  # 清扫宽度 25cm

class CleaningMode(Enum):
    """清扫模式枚举"""
    AUTO = "auto"
    SPOT = "spot"
    EDGE = "edge"
    MANUAL = "manual"

class RobotState(Enum):
    """机器人状态枚举"""
    IDLE = "idle"
    CLEANING = "cleaning"
    CHARGING = "charging"
    ERROR = "error"
    PAUSED = "paused"

# ===== 环境感知模块 =====
class EnvironmentPerception:
    """环境感知模块 - 处理传感器数据并构建环境模型"""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.camera = None
        self.lidar_data = []
        self.environment_map = np.zeros((1000, 1000), dtype=np.uint8)
        self.obstacles = []
        self.setup_sensors()
        
    def setup_sensors(self):
        """初始化传感器"""
        try:
            self.camera = cv2.VideoCapture(self.config.CAMERA_ID)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            logging.info("视觉传感器初始化成功")
        except Exception as e:
            logging.error(f"传感器初始化失败: {e}")
    
    def get_camera_frame(self) -> Optional[np.ndarray]:
        """获取摄像头帧"""
        if self.camera is None:
            return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def detect_obstacles(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测障碍物"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 过滤小的噪声
                x, y, w, h = cv2.boundingRect(contour)
                obstacles.append((x, y, w, h))
        
        return obstacles
    
    def update_environment_map(self, robot_pos: Tuple[float, float], obstacles: List[Tuple[int, int, int, int]]):
        """更新环境地图"""
        # 将障碍物信息更新到环境地图
        for obstacle in obstacles:
            x, y, w, h = obstacle
            # 将像素坐标转换为地图坐标
            map_x = int(x * self.config.MAP_RESOLUTION)
            map_y = int(y * self.config.MAP_RESOLUTION)
            map_w = int(w * self.config.MAP_RESOLUTION)
            map_h = int(h * self.config.MAP_RESOLUTION)
            
            # 在地图上标记障碍物
            self.environment_map[map_y:map_y+map_h, map_x:map_x+map_w] = 255
    
    def get_environment_data(self) -> Dict[str, Any]:
        """获取环境数据"""
        frame = self.get_camera_frame()
        if frame is None:
            return {}
        
        obstacles = self.detect_obstacles(frame)
        
        return {
            'frame': frame,
            'obstacles': obstacles,
            'environment_map': self.environment_map,
            'timestamp': time.time()
        }

# ===== 路径规划模块 =====
class PathPlanning:
    """路径规划模块 - 基于环境模型运用智能算法规划高效清扫路径"""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.current_path = []
        self.cleaning_coverage = np.zeros((1000, 1000), dtype=np.uint8)
        self.robot_position = [500, 500]  # 机器人初始位置
        
    def plan_cleaning_path(self, environment_map: np.ndarray, start_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
        """规划清扫路径 - 使用蛇形路径算法"""
        height, width = environment_map.shape
        path = []
        
        # 蛇形路径规划
        cleaning_width_pixels = int(self.config.CLEANING_WIDTH / self.config.MAP_RESOLUTION)
        
        y = 0
        direction = 1  # 1为向右，-1为向左
        
        while y < height:
            if direction == 1:
                # 向右扫描
                for x in range(0, width, cleaning_width_pixels):
                    if environment_map[y, x] == 0:  # 无障碍物
                        path.append((x * self.config.MAP_RESOLUTION, y * self.config.MAP_RESOLUTION))
            else:
                # 向左扫描
                for x in range(width-1, -1, -cleaning_width_pixels):
                    if environment_map[y, x] == 0:  # 无障碍物
                        path.append((x * self.config.MAP_RESOLUTION, y * self.config.MAP_RESOLUTION))
            
            direction *= -1
            y += cleaning_width_pixels
        
        return path
    
    def calculate_coverage_rate(self) -> float:
        """计算清扫覆盖率"""
        total_area = np.sum(self.cleaning_coverage == 0)  # 可清扫区域
        cleaned_area = np.sum(self.cleaning_coverage == 255)  # 已清扫区域
        
        if total_area == 0:
            return 0.0
        
        return cleaned_area / (total_area + cleaned_area)
    
    def update_cleaning_coverage(self, robot_pos: Tuple[float, float]):
        """更新清扫覆盖情况"""
        map_x = int(robot_pos[0] / self.config.MAP_RESOLUTION)
        map_y = int(robot_pos[1] / self.config.MAP_RESOLUTION)
        
        # 标记已清扫区域
        radius = int(self.config.CLEANING_WIDTH / 2 / self.config.MAP_RESOLUTION)
        cv2.circle(self.cleaning_coverage, (map_x, map_y), radius, 255, -1)
    
    def get_next_waypoint(self) -> Optional[Tuple[float, float]]:
        """获取下一个路径点"""
        if not self.current_path:
            return None
        
        return self.current_path.pop(0)

# ===== 物品识别模块 =====
class ObjectRecognition:
    """物品识别模块 - 利用深度学习算法识别物品"""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.high_value_objects = [
            'laptop', 'phone', 'watch', 'jewelry', 'glasses',
            'camera', 'tablet', 'remote', 'keys', 'wallet'
        ]
        self.recognition_accuracy = 0.0
        self.detection_count = 0
        self.correct_detections = 0
        
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """检测物品 - 简化版本，实际应用中需要使用深度学习模型"""
        # 这里使用简化的检测方法，实际应用中需要集成YOLO、SSD等深度学习模型
        objects = []
        
        # 使用颜色检测模拟物品识别
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 检测金属物品（模拟高价值物品）
        metal_lower = np.array([0, 0, 200])
        metal_upper = np.array([180, 30, 255])
        metal_mask = cv2.inRange(hsv, metal_lower, metal_upper)
        
        contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 过滤小对象
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(0.95, area / 5000)  # 模拟置信度
                
                objects.append({
                    'class': 'high_value_object',
                    'confidence': confidence,
                    'bbox': (x, y, w, h),
                    'is_high_value': True
                })
        
        return objects
    
    def is_high_value_object(self, obj_class: str) -> bool:
        """判断是否为高价值物品"""
        return obj_class in self.high_value_objects
    
    def update_recognition_accuracy(self, predicted: List[str], ground_truth: List[str]):
        """更新识别准确率"""
        if not ground_truth:
            return
        
        self.detection_count += 1
        correct = sum(1 for p, g in zip(predicted, ground_truth) if p == g)
        self.correct_detections += correct
        
        self.recognition_accuracy = self.correct_detections / max(1, self.detection_count)

# ===== 语音交互模块 =====
class VoiceInteraction:
    """语音交互模块 - 处理语音提醒和指令"""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.tts_engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_alerts = Queue()
        self.false_alert_count = 0
        self.total_alerts = 0
        
        # 配置语音引擎
        self.tts_engine.setProperty('rate', 150)  # 语速
        self.tts_engine.setProperty('volume', 0.8)  # 音量
        
    def speak(self, text: str):
        """语音播报"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logging.error(f"语音播报失败: {e}")
    
    def alert_high_value_object(self, obj_type: str, location: Tuple[int, int]):
        """高价值物品提醒"""
        alert_text = f"检测到高价值物品：{obj_type}，位置：{location}，请注意保护"
        self.voice_alerts.put(alert_text)
        self.total_alerts += 1
        
        # 在后台播报
        threading.Thread(target=self.speak, args=(alert_text,), daemon=True).start()
    
    def listen_for_commands(self) -> Optional[str]:
        """监听语音指令"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
            
            command = self.recognizer.recognize_google(audio, language='zh-CN')
            return command.lower()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            logging.error(f"语音识别错误: {e}")
            return None
    
    def get_voice_alert_error_rate(self) -> float:
        """获取语音提醒误报率"""
        if self.total_alerts == 0:
            return 0.0
        return self.false_alert_count / self.total_alerts

# ===== 混合现实交互模块 =====
class MixedRealityInterface:
    """混合现实交互模块 - 提供沉浸式监控与控制体验"""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.ar_overlay = None
        self.response_times = []
        self.last_request_time = 0
        
    def create_ar_overlay(self, frame: np.ndarray, robot_data: Dict[str, Any]) -> np.ndarray:
        """创建AR叠加层"""
        start_time = time.time()
        
        overlay = frame.copy()
        
        # 绘制机器人状态信息
        status_text = f"状态: {robot_data.get('state', 'unknown')}"
        cv2.putText(overlay, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制清扫覆盖率
        coverage = robot_data.get('coverage_rate', 0)
        coverage_text = f"清扫覆盖率: {coverage:.1%}"
        cv2.putText(overlay, coverage_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制电池电量
        battery = robot_data.get('battery_level', 0)
        battery_text = f"电池电量: {battery}%"
        cv2.putText(overlay, battery_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制检测到的高价值物品
        high_value_objects = robot_data.get('high_value_objects', [])
        for i, obj in enumerate(high_value_objects):
            bbox = obj['bbox']
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
            cv2.putText(overlay, f"高价值物品 {obj['confidence']:.2f}", 
                       (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 绘制路径规划
        path = robot_data.get('planned_path', [])
        for i in range(len(path)-1):
            pt1 = (int(path[i][0]), int(path[i][1]))
            pt2 = (int(path[i+1][0]), int(path[i+1][1]))
            cv2.line(overlay, pt1, pt2, (255, 0, 0), 2)
        
        # 计算响应时间
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        # 保持最近100个响应时间记录
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        
        return overlay
    
    def get_average_response_time(self) -> float:
        """获取平均响应时间"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def process_user_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理用户交互"""
        response_start = time.time()
        
        # 处理交互逻辑
        response = {
            'status': 'success',
            'timestamp': time.time(),
            'data': interaction_data
        }
        
        response_time = time.time() - response_start
        self.response_times.append(response_time)
        
        return response

# ===== 硬件控制模块 =====
class HardwareController:
    """硬件控制模块 - 控制电机、清扫组件等硬件"""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.serial_connection = None
        self.motor_speeds = [0, 0]  # 左右电机速度
        self.cleaning_power = 0  # 清扫功率
        self.battery_level = 100  # 电池电量
        self.setup_hardware()
        
    def setup_hardware(self):
        """初始化硬件连接"""
        try:
            self.serial_connection = serial.Serial(
                self.config.SERIAL_PORT, 
                self.config.BAUD_RATE, 
                timeout=1
            )
            logging.info("硬件连接初始化成功")
        except Exception as e:
            logging.warning(f"硬件连接失败，使用模拟模式: {e}")
            self.serial_connection = None
    
    def send_command(self, command: str):
        """发送硬件控制命令"""
        if self.serial_connection:
            try:
                self.serial_connection.write(command.encode())
            except Exception as e:
                logging.error(f"硬件命令发送失败: {e}")
    
    def move_forward(self, speed: float):
        """前进"""
        self.motor_speeds = [speed, speed]
        command = f"MOVE_FORWARD:{speed}\n"
        self.send_command(command)
    
    def move_backward(self, speed: float):
        """后退"""
        self.motor_speeds = [-speed, -speed]
        command = f"MOVE_BACKWARD:{speed}\n"
        self.send_command(command)
    
    def turn_left(self, speed: float):
        """左转"""
        self.motor_speeds = [-speed, speed]
        command = f"TURN_LEFT:{speed}\n"
        self.send_command(command)
    
    def turn_right(self, speed: float):
        """右转"""
        self.motor_speeds = [speed, -speed]
        command = f"TURN_RIGHT:{speed}\n"
        self.send_command(command)
    
    def stop(self):
        """停止"""
        self.motor_speeds = [0, 0]
        command = "STOP\n"
        self.send_command(command)
    
    def start_cleaning(self, power: float):
        """开始清扫"""
        self.cleaning_power = power
        command = f"START_CLEANING:{power}\n"
        self.send_command(command)
    
    def stop_cleaning(self):
        """停止清扫"""
        self.cleaning_power = 0
        command = "STOP_CLEANING\n"
        self.send_command(command)
    
    def get_battery_level(self) -> int:
        """获取电池电量"""
        # 模拟电池电量消耗
        if self.cleaning_power > 0:
            self.battery_level -= 0.01
        
        return max(0, int(self.battery_level))

# ===== 数据存储模块 =====
class DataStorage:
    """数据存储模块 - 存储运行数据和日志"""
    
    def __init__(self, db_path: str = "robot_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建清扫记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cleaning_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                coverage_rate REAL,
                area_cleaned REAL,
                battery_consumed INTEGER,
                objects_detected INTEGER,
                mode TEXT
            )
        ''')
        
        # 创建物品识别记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS object_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                object_type TEXT,
                confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                is_high_value BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_cleaning_record(self, record: Dict[str, Any]):
        """保存清扫记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO cleaning_records 
            (start_time, end_time, coverage_rate, area_cleaned, battery_consumed, objects_detected, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            record['start_time'],
            record['end_time'],
            record['coverage_rate'],
            record['area_cleaned'],
            record['battery_consumed'],
            record['objects_detected'],
            record['mode']
        ))
        
        conn.commit()
        conn.close()
    
    def save_object_detection(self, detection: Dict[str, Any]):
        """保存物品识别记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO object_detections 
            (timestamp, object_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, is_high_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection['timestamp'],
            detection['object_type'],
            detection['confidence'],
            detection['bbox'][0],
            detection['bbox'][1],
            detection['bbox'][2],
            detection['bbox'][3],
            detection['is_high_value']
        ))
        
        conn.commit()
        conn.close()

# ===== 主控制系统 =====
class SmartCleaningRobot:
    """智能清扫机器人主控制系统"""
    
    def __init__(self):
        self.config = RobotConfig()
        self.state = RobotState.IDLE
        self.cleaning_mode = CleaningMode.AUTO
        
        # 初始化各个模块
        self.environment_perception = EnvironmentPerception(self.config)
        self.path_planning = PathPlanning(self.config)
        self.object_recognition = ObjectRecognition(self.config)
        self.voice_interaction = VoiceInteraction(self.config)
        self.mr_interface = MixedRealityInterface(self.config)
        self.hardware_controller = HardwareController(self.config)
        self.data_storage = DataStorage()
        
        # 运行状态
        self.running = False
        self.current_task = None
        self.start_time = None
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
    def start(self):
        """启动机器人"""
        self.running = True
        self.state = RobotState.IDLE
        logging.info("智能清扫机器人启动")
        
        # 启动主循环
        self.main_loop()
    
    def stop(self):
        """停止机器人"""
        self.running = False
        self.state = RobotState.IDLE
        self.hardware_controller.stop()
        self.hardware_controller.stop_cleaning()
        logging.info("智能清扫机器人停止")
    
    def start_cleaning_task(self, mode: CleaningMode = CleaningMode.AUTO):
        """开始清扫任务"""
        self.cleaning_mode = mode
        self.state = RobotState.CLEANING
        self.start_time = datetime.now()
        
        # 获取环境数据
        env_data = self.environment_perception.get_environment_data()
        
        # 规划路径
        if 'environment_map' in env_data:
            path = self.path_planning.plan_cleaning_path(
                env_data['environment_map'], 
                (self.path_planning.robot_position[0], self.path_planning.robot_position[1])
            )
            self.path_planning.current_path = path
        
        # 开始清扫
        self.hardware_controller.start_cleaning(0.8)
        
        logging.info(f"开始清扫任务，模式: {mode.value}")
    
    def pause_cleaning(self):
        """暂停清扫"""
        if self.state == RobotState.CLEANING:
            self.state = RobotState.PAUSED
            self.hardware_controller.stop()
            self.hardware_controller.stop_cleaning()
            logging.info("清扫任务已暂停")
    
    def resume_cleaning(self):
        """恢复清扫"""
        if self.state == RobotState.PAUSED:
            self.state = RobotState.CLEANING
            self.hardware_controller.start_cleaning(0.8)
            logging.info("清扫任务已恢复")
    
    def return_to_charge(self):
        """返回充电"""
        self.state = RobotState.CHARGING
        self.hardware_controller.stop_cleaning()
        # 实际应用中需要实现返回充电桩的路径规划
        logging.info("正在返回充电桩")
    
    def main_loop(self):
        """主循环"""
        while self.running:
            try:
                # 获取环境数据
                env_data = self.environment_perception.get_environment_data()
                
                # 检查电池电量
                battery_level = self.hardware_controller.get_battery_level()
                if battery_level < 20 and self.state != RobotState.CHARGING:
                    self.return_to_charge()
                
                # 根据状态执行相应操作
                if self.state == RobotState.CLEANING:
                    self.execute_cleaning_task(env_data)
                elif self.state == RobotState.IDLE:
                    self.execute_idle_task()
                elif self.state == RobotState.CHARGING:
                    self.execute_charging_task()
                
                # 监听语音指令
                command = self.voice_interaction.listen_for_commands()
                if command:
                    self.process_voice_command(command)
                
                # 更新混合现实界面
                if 'frame' in env_data:
                    robot_data = self.get_robot_status()
                    ar_frame = self.mr_interface.create_ar_overlay(env_data['frame'], robot_data)
                    
                    # 显示AR界面（实际应用中需要发送到用户设备）
                    cv2.imshow('Robot AR Interface', ar_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.1)  # 控制循环频率
                
            except Exception as e:
                logging.error(f"主循环错误: {e}")
                self.state = RobotState.ERROR
                time.sleep(1)
    
    def execute_cleaning_task(self, env_data: Dict[str, Any]):
        """执行清扫任务"""
        if not env_data:
            return
        
        # 物品识别
        if 'frame' in env_data:
            objects = self.object_recognition.detect_objects(env_data['frame'])
            
            # 检查高价值物品
            for obj in objects:
                if obj['is_high_value']:
                    bbox = obj['bbox']
                    self.voice_interaction.alert_high_value_object(
                        obj['class'], 
                        (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                    )
                    
                    # 保存检测记录
                    detection_record = {
                        'timestamp': datetime.now(),
                        'object_type': obj['class'],
                        'confidence': obj['confidence'],
                        'bbox': bbox,
                        'is_high_value': True
                    }
                    self.data_storage.save_object_detection(detection_record)
        
        # 路径规划和移动
        next_waypoint = self.path_planning.get_next_waypoint()
        if next_waypoint:
            self.move_to_waypoint(next_waypoint)
            self.path_planning.update_cleaning_coverage(next_waypoint)
        else:
            # 清扫完成
            self.complete_cleaning_task()
    
    def execute_idle_task(self):
        """执行待机任务"""
        # 在待机状态下进行环境监控
        pass
    
    def execute_charging_task(self):
        """执行充电任务"""
        # 模拟充电过程
        current_battery = self.hardware_controller.get_battery_level()
        if current_battery < 100:
            self.hardware_controller.battery_level += 0.5  # 充电速度
        else:
            self.state = RobotState.IDLE
            logging.info("充电完成")
    
    def move_to_waypoint(self, waypoint: Tuple[float, float]):
        """移动到指定路径点"""
        current_pos = self.path_planning.robot_position
        
        # 计算移动方向
        dx = waypoint[0] - current_pos[0]
        dy = waypoint[1] - current_pos[1]
        
        # 简化的移动控制
        if abs(dx) > abs(dy):
            if dx > 0:
                self.hardware_controller.move_forward(0.5)
            else:
                self.hardware_controller.move_backward(0.5)
        else:
            if dy > 0:
                self.hardware_controller.turn_right(0.5)
            else:
                self.hardware_controller.turn_left(0.5)
        
        # 更新机器人位置
        self.path_planning.robot_position = [waypoint[0], waypoint[1]]
    
    def complete_cleaning_task(self):
        """完成清扫任务"""
        end_time = datetime.now()
        coverage_rate = self.path_planning.calculate_coverage_rate()
        
        # 保存清扫记录
        cleaning_record = {
            'start_time': self.start_time,
            'end_time': end_time,
            'coverage_rate': coverage_rate,
            'area_cleaned': coverage_rate * 100,  # 假设总面积100平方米
            'battery_consumed': 100 - self.hardware_controller.get_battery_level(),
            'objects_detected': len(self.object_recognition.high_value_objects),
            'mode': self.cleaning_mode.value
        }
        
        self.data_storage.save_cleaning_record(cleaning_record)
        
        # 停止清扫
        self.hardware_controller.stop_cleaning()
        self.hardware_controller.stop()
        self.state = RobotState.IDLE
        
        # 语音播报完成
        completion_message = f"清扫任务完成，覆盖率: {coverage_rate:.1%}"
        self.voice_interaction.speak(completion_message)
        
        logging.info(f"清扫任务完成 - 覆盖率: {coverage_rate:.1%}")
    
    def process_voice_command(self, command: str):
        """处理语音指令"""
        command = command.lower()
        
        if "开始清扫" in command or "start cleaning" in command:
            self.start_cleaning_task()
        elif "暂停" in command or "pause" in command:
            self.pause_cleaning()
        elif "继续" in command or "resume" in command:
            self.resume_cleaning()
        elif "停止" in command or "stop" in command:
            self.stop()
        elif "返回充电" in command or "go charge" in command:
            self.return_to_charge()
        elif "状态" in command or "status" in command:
            status = self.get_robot_status_text()
            self.voice_interaction.speak(status)
        else:
            self.voice_interaction.speak("未识别的指令")
    
    def get_robot_status(self) -> Dict[str, Any]:
        """获取机器人状态信息"""
        return {
            'state': self.state.value,
            'cleaning_mode': self.cleaning_mode.value,
            'battery_level': self.hardware_controller.get_battery_level(),
            'coverage_rate': self.path_planning.calculate_coverage_rate(),
            'high_value_objects': [],  # 当前检测到的高价值物品
            'planned_path': self.path_planning.current_path,
            'motor_speeds': self.hardware_controller.motor_speeds,
            'cleaning_power': self.hardware_controller.cleaning_power,
            'voice_error_rate': self.voice_interaction.get_voice_alert_error_rate(),
            'mr_response_time': self.mr_interface.get_average_response_time(),
            'object_recognition_accuracy': self.object_recognition.recognition_accuracy
        }
    
    def get_robot_status_text(self) -> str:
        """获取机器人状态文本"""
        status = self.get_robot_status()
        return f"当前状态: {status['state']}, 电池电量: {status['battery_level']}%, 清扫覆盖率: {status['coverage_rate']:.1%}"
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return {
            'cleaning_coverage_rate': self.path_planning.calculate_coverage_rate(),
            'object_recognition_accuracy': self.object_recognition.recognition_accuracy,
            'voice_alert_error_rate': self.voice_interaction.get_voice_alert_error_rate(),
            'mr_response_time': self.mr_interface.get_average_response_time(),
            'battery_efficiency': self.hardware_controller.get_battery_level() / 100.0
        }


# ===== 云端服务接口 =====
class CloudService:
    """云端服务接口 - 实现数据上传、远程控制等功能"""
    
    def __init__(self, api_endpoint: str = "https://api.volcanicengine.com"):
        self.api_endpoint = api_endpoint
        self.device_id = "smart_robot_001"
        self.api_key = "your_api_key_here"
        
    def upload_cleaning_data(self, data: Dict[str, Any]) -> bool:
        """上传清扫数据到云端"""
        try:
            # 这里应该实现实际的API调用
            # 使用火山引擎的物联网平台API
            payload = {
                'device_id': self.device_id,
                'timestamp': time.time(),
                'data': data
            }
            
            # 模拟API调用
            logging.info(f"上传数据到云端: {payload}")
            return True
            
        except Exception as e:
            logging.error(f"数据上传失败: {e}")
            return False
    
    def get_remote_commands(self) -> List[Dict[str, Any]]:
        """获取远程控制指令"""
        try:
            # 这里应该实现实际的API调用
            # 模拟返回远程指令
            commands = [
                {'command': 'start_cleaning', 'params': {'mode': 'auto'}},
                {'command': 'return_charge', 'params': {}}
            ]
            return commands
            
        except Exception as e:
            logging.error(f"获取远程指令失败: {e}")
            return []
    
    def update_device_status(self, status: Dict[str, Any]) -> bool:
        """更新设备状态到云端"""
        try:
            payload = {
                'device_id': self.device_id,
                'status': status,
                'timestamp': time.time()
            }
            
            logging.info(f"更新设备状态: {payload}")
            return True
            
        except Exception as e:
            logging.error(f"状态更新失败: {e}")
            return False


# ===== 配置文件管理 =====
class ConfigManager:
    """配置文件管理器"""
    
    def __init__(self, config_file: str = "robot_config.json"):
        self.config_file = config_file
        self.default_config = {
            'hardware': {
                'serial_port': '/dev/ttyUSB0',
                'baud_rate': 115200,
                'camera_id': 0
            },
            'cleaning': {
                'default_mode': 'auto',
                'cleaning_power': 0.8,
                'safe_distance': 0.3
            },
            'voice': {
                'language': 'zh-CN',
                'voice_rate': 150,
                'voice_volume': 0.8
            },
            'performance': {
                'target_coverage': 0.9,
                'max_cleaning_time': 3600,
                'battery_low_threshold': 20
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            # 创建默认配置文件
            self.save_config(self.default_config)
            return self.default_config
        except Exception as e:
            logging.error(f"配置文件加载失败: {e}")
            return self.default_config
    
    def save_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"配置文件保存失败: {e}")


# ===== 测试和调试模块 =====
class RobotTester:
    """机器人测试模块"""
    
    def __init__(self, robot: SmartCleaningRobot):
        self.robot = robot
        self.test_results = {}
    
    def run_all_tests(self):
        """运行所有测试"""
        logging.info("开始运行测试套件")
        
        self.test_hardware_connection()
        self.test_vision_system()
        self.test_path_planning()
        self.test_object_recognition()
        self.test_voice_interaction()
        self.test_mr_interface()
        
        self.generate_test_report()
    
    def test_hardware_connection(self):
        """测试硬件连接"""
        try:
            # 测试电机控制
            self.robot.hardware_controller.move_forward(0.1)
            time.sleep(0.5)
            self.robot.hardware_controller.stop()
            
            self.test_results['hardware'] = {
                'status': 'PASS',
                'details': '硬件连接正常'
            }
        except Exception as e:
            self.test_results['hardware'] = {
                'status': 'FAIL',
                'details': f'硬件连接测试失败: {e}'
            }
    
    def test_vision_system(self):
        """测试视觉系统"""
        try:
            frame = self.robot.environment_perception.get_camera_frame()
            if frame is not None:
                obstacles = self.robot.environment_perception.detect_obstacles(frame)
                self.test_results['vision'] = {
                    'status': 'PASS',
                    'details': f'检测到 {len(obstacles)} 个障碍物'
                }
            else:
                self.test_results['vision'] = {
                    'status': 'FAIL',
                    'details': '无法获取摄像头图像'
                }
        except Exception as e:
            self.test_results['vision'] = {
                'status': 'FAIL',
                'details': f'视觉系统测试失败: {e}'
            }
    
    def test_path_planning(self):
        """测试路径规划"""
        try:
            # 创建测试环境地图
            test_map = np.zeros((100, 100), dtype=np.uint8)
            path = self.robot.path_planning.plan_cleaning_path(test_map, (0, 0))
            
            if len(path) > 0:
                self.test_results['path_planning'] = {
                    'status': 'PASS',
                    'details': f'生成路径包含 {len(path)} 个点'
                }
            else:
                self.test_results['path_planning'] = {
                    'status': 'FAIL',
                    'details': '路径规划失败'
                }
        except Exception as e:
            self.test_results['path_planning'] = {
                'status': 'FAIL',
                'details': f'路径规划测试失败: {e}'
            }
    
    def test_object_recognition(self):
        """测试物品识别"""
        try:
            # 创建测试图像
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            objects = self.robot.object_recognition.detect_objects(test_frame)
            
            self.test_results['object_recognition'] = {
                'status': 'PASS',
                'details': f'物品识别系统工作正常，检测到 {len(objects)} 个物品'
            }
        except Exception as e:
            self.test_results['object_recognition'] = {
                'status': 'FAIL',
                'details': f'物品识别测试失败: {e}'
            }
    
    def test_voice_interaction(self):
        """测试语音交互"""
        try:
            # 测试语音播报
            self.robot.voice_interaction.speak("测试语音播报")
            
            self.test_results['voice_interaction'] = {
                'status': 'PASS',
                'details': '语音交互系统正常'
            }
        except Exception as e:
            self.test_results['voice_interaction'] = {
                'status': 'FAIL',
                'details': f'语音交互测试失败: {e}'
            }
    
    def test_mr_interface(self):
        """测试混合现实接口"""
        try:
            # 创建测试数据
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            test_data = {'state': 'test', 'coverage_rate': 0.5}
            
            ar_frame = self.robot.mr_interface.create_ar_overlay(test_frame, test_data)
            response_time = self.robot.mr_interface.get_average_response_time()
            
            self.test_results['mr_interface'] = {
                'status': 'PASS',
                'details': f'混合现实接口正常，平均响应时间: {response_time:.3f}s'
            }
        except Exception as e:
            self.test_results['mr_interface'] = {
                'status': 'FAIL',
                'details': f'混合现实接口测试失败: {e}'
            }
    
    def generate_test_report(self):
        """生成测试报告"""
        report = "=" * 50 + "\n"
        report += "智能清扫机器人测试报告\n"
        report += "=" * 50 + "\n\n"
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        
        report += f"测试总数: {total_tests}\n"
        report += f"通过测试: {passed_tests}\n"
        report += f"失败测试: {total_tests - passed_tests}\n"
        report += f"通过率: {passed_tests/total_tests*100:.1f}%\n\n"
        
        for test_name, result in self.test_results.items():
            status_symbol = "✓" if result['status'] == 'PASS' else "✗"
            report += f"{status_symbol} {test_name}: {result['details']}\n"
        
        report += "\n" + "=" * 50
        
        print(report)
        
        # 保存测试报告
        with open('test_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)


# ===== 主程序入口 =====
def main():
    """主程序入口"""
    try:
        # 创建机器人实例
        robot = SmartCleaningRobot()
        
        # 运行测试（可选）
        if len(sys.argv) > 1 and sys.argv[1] == '--test':
            tester = RobotTester(robot)
            tester.run_all_tests()
            return
        
        # 启动机器人
        print("正在启动基于RDK X5的视觉智能调度车...")
        print("按 Ctrl+C 停止机器人")
        
        robot.start()
        
    except KeyboardInterrupt:
        print("\n正在停止机器人...")
        robot.stop()
        print("机器人已停止")
    except Exception as e:
        logging.error(f"程序运行错误: {e}")
        print(f"程序运行错误: {e}")


if __name__ == "__main__":
    import sys
    main()


# ===== 附加工具脚本 =====

# 性能监控脚本
"""
performance_monitor.py - 性能监控工具

使用方法:
python performance_monitor.py

功能:
- 实时监控机器人性能指标
- 生成性能报告
- 性能数据可视化
"""

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, robot: SmartCleaningRobot):
        self.robot = robot
        self.metrics_history = []
        self.monitoring = False
        
    def start_monitoring(self, interval: float = 1.0):
        """开始性能监控"""
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                metrics = self.robot.get_performance_metrics()
                metrics['timestamp'] = time.time()
                self.metrics_history.append(metrics)
                
                # 保持最近1000条记录
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)
                
                time.sleep(interval)
        
        threading.Thread(target=monitor_loop, daemon=True).start()
        logging.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        logging.info("性能监控已停止")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {}
        
        # 计算平均值
        avg_metrics = {}
        for key in self.metrics_history[0].keys():
            if key != 'timestamp':
                values = [m[key] for m in self.metrics_history if key in m]
                avg_metrics[f'avg_{key}'] = sum(values) / len(values)
        
        return {
            'total_samples': len(self.metrics_history),
            'monitoring_duration': self.metrics_history[-1]['timestamp'] - self.metrics_history[0]['timestamp'],
            'average_metrics': avg_metrics
        }


# 数据分析脚本
"""
data_analyzer.py - 数据分析工具

使用方法:
python data_analyzer.py

功能:
- 分析清扫数据
- 生成统计报告
- 优化建议
"""

class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, db_path: str = "robot_data.db"):
        self.db_path = db_path
    
    def analyze_cleaning_efficiency(self) -> Dict[str, Any]:
        """分析清扫效率"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取清扫记录
        cursor.execute("""
            SELECT coverage_rate, area_cleaned, battery_consumed, 
                   (julianday(end_time) - julianday(start_time)) * 24 * 60 as duration_minutes
            FROM cleaning_records
            WHERE end_time IS NOT NULL
        """)
        
        records = cursor.fetchall()
        conn.close()
        
        if not records:
            return {'error': '没有清扫记录'}
        
        # 计算统计数据
        coverage_rates = [r[0] for r in records]
        areas_cleaned = [r[1] for r in records]
        battery_consumptions = [r[2] for r in records]
        durations = [r[3] for r in records]
        
        return {
            'total_cleanings': len(records),
            'avg_coverage_rate': sum(coverage_rates) / len(coverage_rates),
            'avg_area_cleaned': sum(areas_cleaned) / len(areas_cleaned),
            'avg_battery_consumption': sum(battery_consumptions) / len(battery_consumptions),
            'avg_duration_minutes': sum(durations) / len(durations),
            'efficiency_score': (sum(coverage_rates) / len(coverage_rates)) / (sum(battery_consumptions) / len(battery_consumptions))
        }
    
    def generate_optimization_suggestions(self) -> List[str]:
        """生成优化建议"""
        analysis = self.analyze_cleaning_efficiency()
        suggestions = []
        
        if analysis.get('avg_coverage_rate', 0) < 0.9:
            suggestions.append("建议优化路径规划算法，提高清扫覆盖率")
        
        if analysis.get('avg_battery_consumption', 0) > 80:
            suggestions.append("建议优化电源管理，降低电池消耗")
        
        if analysis.get('avg_duration_minutes', 0) > 120:
            suggestions.append("建议优化清扫策略，缩短清扫时间")
        
        return suggestions


# 远程控制脚本
"""
remote_controller.py - 远程控制工具

使用方法:
python remote_controller.py

功能:
- 远程控制机器人
- 查看实时状态
- 发送控制指令
"""

class RemoteController:
    """远程控制器"""
    
    def __init__(self, robot: SmartCleaningRobot):
        self.robot = robot
        self.cloud_service = CloudService()
    
    def send_command(self, command: str, params: Dict[str, Any] = None):
        """发送远程控制指令"""
        try:
            if command == 'start_cleaning':
                mode = params.get('mode', 'auto')
                self.robot.start_cleaning_task(CleaningMode(mode))
            elif command == 'stop_cleaning':
                self.robot.stop()
            elif command == 'pause_cleaning':
                self.robot.pause_cleaning()
            elif command == 'resume_cleaning':
                self.robot.resume_cleaning()
            elif command == 'return_charge':
                self.robot.return_to_charge()
            
            return {'status': 'success', 'message': f'指令 {command} 执行成功'}
        
        except Exception as e:
            return {'status': 'error', 'message': f'指令执行失败: {e}'}
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """获取实时状态"""
        return self.robot.get_robot_status()
    
    def upload_status_to_cloud(self):
        """上传状态到云端"""
        status = self.get_real_time_status()
        return self.cloud_service.update_device_status(status)


# 配置示例
"""
robot_config.json - 配置文件示例

{
  "hardware": {
    "serial_port": "/dev/ttyUSB0",
    "baud_rate": 115200,
    "camera_id": 0
  },
  "cleaning": {
    "default_mode": "auto",
    "cleaning_power": 0.8,
    "safe_distance": 0.3
  },
  "voice": {
    "language": "zh-CN",
    "voice_rate": 150,
    "voice_volume": 0.8
  },
  "performance": {
    "target_coverage": 0.9,
    "max_cleaning_time": 3600,
    "battery_low_threshold": 20
  }
}
"""

# 安装依赖包的requirements.txt
"""
requirements.txt

opencv-python==4.8.0.74
numpy==1.24.3
pyserial==3.5
pyttsx3==2.90
SpeechRecognition==3.10.0
sqlite3
threading
logging
dataclasses
enum34
queue
datetime
json
time
os
sys
"""
