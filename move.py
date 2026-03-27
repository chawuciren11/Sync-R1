import os
import shutil
from pathlib import Path

def move_concept_files():
    # 定义源目录和目标目录
    source_base = Path("output/test")
    target_base = Path("dataset/unictokens_data/concept/test")
    
    # 检查源目录是否存在
    if not source_base.exists():
        print(f"错误: 源目录不存在 {source_base}")
        return False
    
    # 检查目标目录是否存在
    if not target_base.exists():
        print(f"错误: 目标目录不存在 {target_base}")
        return False
    
    # 遍历源目录中的所有concept文件夹
    moved_count = 0
    skipped_count = 0
    
    for concept_dir in source_base.iterdir():
        if concept_dir.is_dir():
            concept_name = concept_dir.name
            source_info_file = concept_dir / "t2i_conditions.json"
            target_concept_dir = target_base / concept_name
            target_info_file = target_concept_dir / "t2i_conditions.json"
            
            # 检查源t2i_conditions.json文件是否存在
            if not source_info_file.exists():
                print(f"跳过: {concept_name} - 源t2i_conditions.json不存在")
                skipped_count += 1
                continue
            
            # 检查目标目录是否存在
            if not target_concept_dir.exists():
                print(f"跳过: {concept_name} - 目标目录不存在")
                skipped_count += 1
                continue
            
            # 检查目标t2i_conditions.json文件是否存在（可选，用于确认）
            if not target_info_file.exists():
                print(f"警告: {concept_name} - 目标t2i_conditions.json不存在，将创建新文件")
            
            try:
                # 复制文件（覆盖目标文件）
                shutil.copy2(source_info_file, target_info_file)
                print(f"成功: {concept_name} - t2i_conditions.json 已覆盖")
                moved_count += 1
                
            except Exception as e:
                print(f"错误: {concept_name} - 文件移动失败: {e}")
                skipped_count += 1
    
    # 输出总结
    print(f"\n操作完成:")
    print(f"成功移动: {moved_count} 个文件")
    print(f"跳过: {skipped_count} 个目录")
    
    return moved_count > 0

def move_concept_files_with_validation():
    """
    带验证的版本，先显示将要执行的操作，确认后再执行
    """
    source_base = Path("output/test")
    target_base = Path("dataset/unictokens_data/concept/test")
    
    if not source_base.exists() or not target_base.exists():
        print("源目录或目标目录不存在")
        return False
    
    # 首先收集所有要移动的文件信息
    operations = []
    
    for concept_dir in source_base.iterdir():
        if concept_dir.is_dir():
            concept_name = concept_dir.name
            source_info_file = concept_dir / "t2i_conditions.json"
            target_concept_dir = target_base / concept_name
            target_info_file = target_concept_dir / "t2i_conditions.json"
            
            if source_info_file.exists() and target_concept_dir.exists():
                operations.append({
                    'concept': concept_name,
                    'source': source_info_file,
                    'target': target_info_file,
                    'target_exists': target_info_file.exists()
                })
    
    if not operations:
        print("没有找到需要移动的文件")
        return False
    
    # 显示将要执行的操作
    print("将要执行以下操作:")
    print("-" * 60)
    for op in operations:
        status = "覆盖" if op['target_exists'] else "创建"
        print(f"{status}: {op['concept']}")
        print(f"  源: {op['source']}")
        print(f"  目标: {op['target']}")
        print()
    
    # 请求用户确认
    confirm = input(f"确认执行以上 {len(operations)} 个操作? (y/N): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return False
    
    # 执行移动操作
    success_count = 0
    for op in operations:
        try:
            # 确保目标目录存在
            op['target'].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(op['source'], op['target'])
            print(f"✓ 成功: {op['concept']}")
            success_count += 1
        except Exception as e:
            print(f"✗ 失败: {op['concept']} - {e}")
    
    print(f"\n操作完成: 成功 {success_count}/{len(operations)}")
    return success_count > 0

def dry_run():
    """
    干运行模式：只显示将要执行的操作，不实际移动文件
    """
    source_base = Path("output/test")
    target_base = Path("dataset/unictokens_data/concept/test")
    
    print("干运行模式 - 显示将要执行的操作:")
    print("=" * 60)
    
    operations = []
    for concept_dir in source_base.iterdir():
        if concept_dir.is_dir():
            concept_name = concept_dir.name
            source_info_file = concept_dir / "t2i_conditions.json"
            target_concept_dir = target_base / concept_name
            
            if source_info_file.exists() and target_concept_dir.exists():
                target_info_file = target_concept_dir / "t2i_conditions.json"
                operations.append({
                    'concept': concept_name,
                    'source': source_info_file,
                    'target': target_info_file,
                    'target_exists': target_info_file.exists()
                })
    
    if not operations:
        print("没有找到需要移动的文件")
        return
    
    for op in operations:
        status = "覆盖" if op['target_exists'] else "创建"
        print(f"{status}: {op['concept']}")
        print(f"  源: {op['source']}")
        print(f"  目标: {op['target']}")
    
    print(f"\n总计: {len(operations)} 个操作")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='移动concept的t2i_conditions.json文件')
    parser.add_argument('--dry-run', action='store_true', help='干运行模式，只显示不执行')
    parser.add_argument('--confirm', action='store_true', help='需要确认后再执行')
    
    args = parser.parse_args()
    
    if args.dry_run:
        dry_run()
    elif args.confirm:
        move_concept_files_with_validation()
    else:
        move_concept_files()