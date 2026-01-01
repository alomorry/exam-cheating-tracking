import socket
import json

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 8080))
    server.listen(5)
    print("Сервер запущен, ожидание подключений...")

    while True:
        client, addr = server.accept()
        try:
            data = client.recv(1024).decode('utf-8')
            if data:
                event = json.loads(data)
                print(f"[Алерт от {addr}] Время: {event['time_sec']}с | Тип: {event['event']} | Детали: {event['info']}")
        except Exception as e:
            print(f"Ошибка при получении данных: {e}")
        finally:
            client.close()

if __name__ == "__main__":
    start_server()