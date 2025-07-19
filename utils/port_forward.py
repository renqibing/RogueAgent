import sys
import socket
import select
from threading import Thread
import time

def handle_client(client_socket, target_port):
    # 为vLLM设置更长的超时时间
    client_socket.settimeout(3600)  # 1小时超时，适应长时间运行的推理任务
    
    try:
        target_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        target_socket.settimeout(3600)  # 匹配客户端超时
        target_socket.connect(('localhost', target_port))
        
        def forward(src, dst, name):
            try:
                while True:
                    try:
                        # 使用select进行非阻塞读取，但等待更长时间
                        ready, _, _ = select.select([src], [], [], 600)  # 10分钟等待时间
                        if not ready:
                            # 对于长时间无数据的情况，只是继续等待而不是断开
                            continue
                            
                        data = src.recv(8192)  # 增大缓冲区大小以处理更大的响应
                        if not data:
                            break
                        dst.send(data)
                    except socket.timeout:
                        # 对于超时，我们不立即断开，而是继续尝试
                        continue
                    except ConnectionResetError:
                        # 连接被重置，需要退出循环
                        break
                    except BrokenPipeError:
                        # 管道损坏，需要退出循环
                        break
            except Exception as e:
                print(f"Error in {name} forward: {str(e)}")
            finally:
                # 静默关闭连接
                try:
                    src.close()
                except:
                    pass
                try:
                    dst.close()
                except:
                    pass
        
        # 创建两个转发线程
        client_to_target = Thread(target=forward, args=(client_socket, target_socket, "client->vLLM"))
        target_to_client = Thread(target=forward, args=(target_socket, client_socket, "vLLM->client"))
        
        # 设置为后台线程
        client_to_target.daemon = True
        target_to_client.daemon = True
        
        # 启动线程
        client_to_target.start()
        target_to_client.start()
        
        # 没有固定的join超时，让线程持续运行直到连接自然结束
        client_to_target.join()
        target_to_client.join()
        
    except Exception as e:
        print(f"Connection setup error: {str(e)}")
        try:
            client_socket.close()
        except:
            pass

def start_proxy(listen_port, target_port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 调整TCP保活参数，适用于长连接
    server.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    
    # 在支持的平台上，可以进一步调整TCP保活参数
    # 这些参数在不同平台可能需要不同设置
    try:
        # TCP_KEEPIDLE: 空闲多久后开始发送保活探测包
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 300)
        # TCP_KEEPINTVL: 保活探测包之间的间隔
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60)
        # TCP_KEEPCNT: 尝试发送多少次探测包后认为连接断开
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 8)
    except (AttributeError, OSError):
        # 忽略不支持的平台的错误
        pass
    
    try:
        server.bind(('0.0.0.0', listen_port))
        server.listen(20)  # 增加队列大小以处理更多并发连接
        print(f"Forwarding {listen_port} -> {target_port}")
        
        # 设置服务器套接字超时，但保持较长超时
        server.settimeout(3.0)
        
        running = True
        while running:
            try:
                client, addr = server.accept()
                print(f"New connection from {addr[0]}:{addr[1]}")
                
                # 为每个客户端创建新线程
                handler = Thread(target=handle_client, args=(client, target_port))
                handler.daemon = True
                handler.start()
            except socket.timeout:
                # 短暂超时，继续循环
                continue
            except KeyboardInterrupt:
                print("Received interrupt, shutting down...")
                running = False
            except Exception as e:
                print(f"Accept error: {str(e)}")
                # 避免在错误情况下快速循环消耗CPU
                time.sleep(0.5)
    except Exception as e:
        print(f"Server error: {str(e)}")
    finally:
        print("Proxy server shutting down")
        server.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python proxy.py [listen_port] [target_port]")
        sys.exit(1)
    try:
        start_proxy(int(sys.argv[1]), int(sys.argv[2]))
    except KeyboardInterrupt:
        print("Proxy terminated by user")