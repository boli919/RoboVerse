#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
URDF到XML转换器
支持多种转换模式：
1. 直接复制URDF内容（因为URDF本身就是XML）
2. 提取关键信息并生成简化的XML
3. 转换为自定义XML结构
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import Dict, List, Optional
import os


class URDFToXMLConverter:
    def __init__(self):
        self.robot_data = {}

    def load_urdf(self, urdf_file_path: str) -> bool:
        """
        加载URDF文件

        Args:
            urdf_file_path: URDF文件路径

        Returns:
            bool: 加载是否成功
        """
        try:
            self.tree = ET.parse(urdf_file_path)
            self.root = self.tree.getroot()
            print(f"成功加载URDF文件: {urdf_file_path}")
            return True
        except Exception as e:
            print(f"加载URDF文件失败: {e}")
            return False

    def extract_robot_info(self) -> Dict:
        """
        提取机器人基本信息

        Returns:
            Dict: 机器人信息字典
        """
        robot_info = {
            'name': self.root.get('name', 'unknown'),
            'links': [],
            'joints': [],
            'materials': []
        }

        # 提取链接信息
        for link in self.root.findall('link'):
            link_info = {
                'name': link.get('name', ''),
                'visual': self._extract_visual_info(link),
                'collision': self._extract_collision_info(link),
                'inertial': self._extract_inertial_info(link)
            }
            robot_info['links'].append(link_info)

        # 提取关节信息
        for joint in self.root.findall('joint'):
            joint_info = {
                'name': joint.get('name', ''),
                'type': joint.get('type', ''),
                'parent': self._get_joint_parent(joint),
                'child': self._get_joint_child(joint),
                'origin': self._extract_origin_info(joint),
                'axis': self._extract_axis_info(joint),
                'limit': self._extract_limit_info(joint)
            }
            robot_info['joints'].append(joint_info)

        # 提取材质信息
        for material in self.root.findall('material'):
            material_info = {
                'name': material.get('name', ''),
                'color': self._extract_color_info(material)
            }
            robot_info['materials'].append(material_info)

        return robot_info

    def _extract_visual_info(self, link) -> Dict:
        """提取视觉信息"""
        visual = link.find('visual')
        if visual is None:
            return {}

        info = {}

        # 几何信息
        geometry = visual.find('geometry')
        if geometry is not None:
            info['geometry'] = self._extract_geometry_info(geometry)

        # 材质信息
        material = visual.find('material')
        if material is not None:
            info['material'] = material.get('name', '')

        # 原点信息
        origin = visual.find('origin')
        if origin is not None:
            info['origin'] = self._extract_origin_from_element(origin)

        return info

    def _extract_collision_info(self, link) -> Dict:
        """提取碰撞信息"""
        collision = link.find('collision')
        if collision is None:
            return {}

        info = {}

        # 几何信息
        geometry = collision.find('geometry')
        if geometry is not None:
            info['geometry'] = self._extract_geometry_info(geometry)

        # 原点信息
        origin = collision.find('origin')
        if origin is not None:
            info['origin'] = self._extract_origin_from_element(origin)

        return info

    def _extract_inertial_info(self, link) -> Dict:
        """提取惯性信息"""
        inertial = link.find('inertial')
        if inertial is None:
            return {}

        info = {}

        # 质量
        mass = inertial.find('mass')
        if mass is not None:
            info['mass'] = mass.get('value', '0')

        # 惯性矩阵
        inertia = inertial.find('inertia')
        if inertia is not None:
            info['inertia'] = {
                'ixx': inertia.get('ixx', '0'),
                'ixy': inertia.get('ixy', '0'),
                'ixz': inertia.get('ixz', '0'),
                'iyy': inertia.get('iyy', '0'),
                'iyz': inertia.get('iyz', '0'),
                'izz': inertia.get('izz', '0')
            }

        return info

    def _extract_geometry_info(self, geometry) -> Dict:
        """提取几何信息"""
        info = {}

        # 检查几何类型
        for geo_type in ['box', 'cylinder', 'sphere', 'mesh']:
            element = geometry.find(geo_type)
            if element is not None:
                info['type'] = geo_type
                if geo_type == 'box':
                    info['size'] = element.get('size', '')
                elif geo_type == 'cylinder':
                    info['radius'] = element.get('radius', '')
                    info['length'] = element.get('length', '')
                elif geo_type == 'sphere':
                    info['radius'] = element.get('radius', '')
                elif geo_type == 'mesh':
                    info['filename'] = element.get('filename', '')
                    info['scale'] = element.get('scale', '1 1 1')
                break

        return info

    def _extract_origin_info(self, element) -> Dict:
        """提取原点信息"""
        origin = element.find('origin')
        return self._extract_origin_from_element(origin) if origin is not None else {}

    def _extract_origin_from_element(self, origin) -> Dict:
        """从origin元素提取信息"""
        return {
            'xyz': origin.get('xyz', '0 0 0'),
            'rpy': origin.get('rpy', '0 0 0')
        }

    def _extract_axis_info(self, joint) -> Dict:
        """提取轴信息"""
        axis = joint.find('axis')
        return {'xyz': axis.get('xyz', '0 0 1')} if axis is not None else {}

    def _extract_limit_info(self, joint) -> Dict:
        """提取限制信息"""
        limit = joint.find('limit')
        if limit is None:
            return {}

        return {
            'lower': limit.get('lower', ''),
            'upper': limit.get('upper', ''),
            'effort': limit.get('effort', ''),
            'velocity': limit.get('velocity', '')
        }

    def _extract_color_info(self, material) -> Dict:
        """提取颜色信息"""
        color = material.find('color')
        return {'rgba': color.get('rgba', '1 1 1 1')} if color is not None else {}

    def _get_joint_parent(self, joint) -> str:
        """获取关节父链接"""
        parent = joint.find('parent')
        return parent.get('link', '') if parent is not None else ''

    def _get_joint_child(self, joint) -> str:
        """获取关节子链接"""
        child = joint.find('child')
        return child.get('link', '') if child is not None else ''

    def convert_to_simple_xml(self, output_file: str) -> bool:
        """
        转换为简化的XML格式

        Args:
            output_file: 输出文件路径

        Returns:
            bool: 转换是否成功
        """
        try:
            robot_info = self.extract_robot_info()

            # 创建根元素
            root = ET.Element('robot_description')
            root.set('name', robot_info['name'])

            # 添加链接信息
            links_element = ET.SubElement(root, 'links')
            for link in robot_info['links']:
                link_element = ET.SubElement(links_element, 'link')
                link_element.set('name', link['name'])

                # 添加视觉信息
                if link['visual']:
                    visual_element = ET.SubElement(link_element, 'visual')
                    self._add_dict_to_element(visual_element, link['visual'])

                # 添加碰撞信息
                if link['collision']:
                    collision_element = ET.SubElement(link_element, 'collision')
                    self._add_dict_to_element(collision_element, link['collision'])

                # 添加惯性信息
                if link['inertial']:
                    inertial_element = ET.SubElement(link_element, 'inertial')
                    self._add_dict_to_element(inertial_element, link['inertial'])

            # 添加关节信息
            joints_element = ET.SubElement(root, 'joints')
            for joint in robot_info['joints']:
                joint_element = ET.SubElement(joints_element, 'joint')
                joint_element.set('name', joint['name'])
                joint_element.set('type', joint['type'])

                if joint['parent']:
                    parent_element = ET.SubElement(joint_element, 'parent')
                    parent_element.text = joint['parent']

                if joint['child']:
                    child_element = ET.SubElement(joint_element, 'child')
                    child_element.text = joint['child']

                # 添加其他关节信息
                for key, value in joint.items():
                    if key not in ['name', 'type', 'parent', 'child'] and value:
                        if isinstance(value, dict):
                            sub_element = ET.SubElement(joint_element, key)
                            self._add_dict_to_element(sub_element, value)

            # 添加材质信息
            if robot_info['materials']:
                materials_element = ET.SubElement(root, 'materials')
                for material in robot_info['materials']:
                    material_element = ET.SubElement(materials_element, 'material')
                    material_element.set('name', material['name'])
                    if material['color']:
                        color_element = ET.SubElement(material_element, 'color')
                        self._add_dict_to_element(color_element, material['color'])

            # 格式化并保存
            self._save_formatted_xml(root, output_file)
            print(f"简化XML已保存到: {output_file}")
            return True

        except Exception as e:
            print(f"转换为简化XML失败: {e}")
            return False

    def _add_dict_to_element(self, element, data_dict):
        """将字典数据添加到XML元素"""
        for key, value in data_dict.items():
            if isinstance(value, dict):
                sub_element = ET.SubElement(element, key)
                self._add_dict_to_element(sub_element, value)
            else:
                if value:  # 只添加非空值
                    element.set(key, str(value))

    def copy_urdf_as_xml(self, output_file: str) -> bool:
        """
        直接复制URDF内容为XML（因为URDF本身就是XML）

        Args:
            output_file: 输出文件路径

        Returns:
            bool: 复制是否成功
        """
        try:
            self._save_formatted_xml(self.root, output_file)
            print(f"URDF已复制为XML: {output_file}")
            return True
        except Exception as e:
            print(f"复制URDF为XML失败: {e}")
            return False

    def _save_formatted_xml(self, root, output_file):
        """保存格式化的XML文件"""
        # 创建格式化的XML字符串
        rough_string = ET.tostring(root, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        formatted_xml = reparsed.toprettyxml(indent="  ", encoding='utf-8')

        # 保存到文件
        with open(output_file, 'wb') as f:
            f.write(formatted_xml)


def main():
    """主函数 - 使用示例"""
    converter = URDFToXMLConverter()

    # 输入文件路径（请修改为你的URDF文件路径）
    urdf_file = "roboverse_data/robots/g1/urdf/g1.urdf"  # 修改为实际的URDF文件路径

    # 检查文件是否存在
    if not os.path.exists(urdf_file):
        print(f"URDF文件不存在: {urdf_file}")
        print("请修改main()函数中的urdf_file变量为正确的文件路径")
        return

    # 加载URDF文件
    if not converter.load_urdf(urdf_file):
        return

    # 转换选项
    print("\n选择转换方式:")
    print("1. 直接复制为XML（保持原始URDF结构）")
    print("2. 转换为简化的XML结构")
    print("3. 两种方式都执行")

    choice = input("请输入选择 (1/2/3): ").strip()

    if choice in ['1', '3']:
        # 直接复制URDF为XML
        output_file1 = urdf_file.replace('.urdf', '_copy.xml')
        converter.copy_urdf_as_xml(output_file1)

    if choice in ['2', '3']:
        # 转换为简化XML
        output_file2 = urdf_file.replace('.urdf', '_simplified.xml')
        converter.convert_to_simple_xml(output_file2)

    print("\n转换完成！")


if __name__ == "__main__":
    main()
