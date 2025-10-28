import os  
import threading  
  
# 检查系统线程限制  
max_users = os.sysconf('SC_ARG_MAX')  
print(f"系统最大线程数限制: {max_users}")  
  
# 获取当前线程数  
current_threads = threading.active_count()  
print(f"当前活跃线程数: {current_threads}")  
  
# 尝试创建新线程  
try:  
    import os  
    
    # 设置OpenBLAS线程数  
    os.environ['OPENBLAS_NUM_THREADS'] = '4'  
    print(f"OPENBLAS_NUM_THREADS set to 4")  
    
    # 尝试创建线程测试  
    import threading  
    threads = []  
    try:  
        for _ in range(10000):  
            t = threading.Thread(target=lambda: None)  
            t.start()  
            threads.append(t)  
        print(f"Successfully created {len(threads)} threads")  
    except RuntimeError as e:  
        print(f"Thread creation failed: {e}")  
    finally:  
        for t in threads:  
            t.join()
    
except RuntimeError as e:  
    print(f"线程创建失败: {e}")
