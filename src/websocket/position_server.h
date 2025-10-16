#ifndef POSITION_SERVER_H
#define POSITION_SERVER_H

#include <memory>
#include <thread>
#include <atomic>
#include <vector>
#include <map>
#include <mutex>
#include <string>
#include <chrono>
#include "../types/yolo_datatype.h"

// 前向声明
struct PersonPosition;

// 人员位置数据结构
struct PositionData {
    int person_id;       // 检测到的人员编号 (从0开始)
    float x_normalized;  // 归一化的X坐标 (-1 到 1)
    float y_normalized;  // 归一化的Y坐标 (固定为0)
    std::map<int, KeyPoint> keypoints;  // 所有关键点信息
    bool valid;          // 数据是否有效
    bool stable;         // 是否稳定（当前项目总是true）

    PositionData(int id = 0, float x = 0.0f, float y = 0.0f, bool v = false)
        : person_id(id), x_normalized(x), y_normalized(y), valid(v), stable(true) {}
};

// 客户端连接管理结构
struct ClientConnection {
    int socket_fd;
    std::chrono::steady_clock::time_point last_ping;
    std::chrono::steady_clock::time_point last_data_sent;
    int failure_count;
    bool is_active;
    std::string client_ip;

    ClientConnection(int fd = -1)
        : socket_fd(fd), last_ping(std::chrono::steady_clock::now()),
          last_data_sent(std::chrono::steady_clock::now()),
          failure_count(0), is_active(true) {}
};

class PositionServer {
public:
    PositionServer(const std::string& host = "0.0.0.0", int port = 3000);
    ~PositionServer();

    // 启动和停止服务器
    bool Start();
    void Stop();
    bool IsRunning() const { return is_running_; }

    // 更新人员位置数据
    void UpdatePositionData(const std::vector<PositionData>& positions);

    // 从PersonPosition和keypoints结构转换数据
    static std::vector<PositionData> ExtractFromPersonPositions(
        const std::vector<PersonPosition>& person_positions,
        const std::vector<std::map<int, KeyPoint>>& keypoints);

private:
    // 服务器配置
    std::string host_;
    int port_;

    // 服务器状态
    std::atomic<bool> is_running_;
    std::atomic<bool> should_stop_;

    // 数据存储和同步
    std::vector<PositionData> position_data_;
    mutable std::mutex data_mutex_;

    // 客户端连接管理
    std::vector<std::unique_ptr<ClientConnection>> clients_;
    std::mutex clients_mutex_;
    std::atomic<int> max_clients_{10};

    // 线程管理
    std::unique_ptr<std::thread> server_thread_;
    std::unique_ptr<std::thread> health_check_thread_;
    std::unique_ptr<std::thread> data_broadcast_thread_;

    // 健壮性参数
    static constexpr int MAX_FAILURES = 3;
    static constexpr int PING_INTERVAL_SECONDS = 5;
    static constexpr int HEALTH_CHECK_INTERVAL_MS = 1000;
    static constexpr int BROADCAST_INTERVAL_MS = 50; // ~20 FPS
    static constexpr int CLIENT_TIMEOUT_SECONDS = 15;

    // WebSocket实现
    void ServerThreadFunc();
    void HandleNewClient(int client_socket, const std::string& client_ip);
    void HealthCheckThreadFunc();
    void DataBroadcastThreadFunc();
    void CleanupDisconnectedClients();
    bool ValidateClient(ClientConnection& client);
    void BroadcastToAllClients(const std::string& data);

    std::string GenerateWebSocketResponse(const std::string& key);
    std::string CreatePositionDataJSON();
    bool SendWebSocketFrame(int socket, const std::string& data);
    bool IsWebSocketUpgrade(const std::string& request);
    std::string ExtractWebSocketKey(const std::string& request);
    std::string GetClientIP(int socket);

    // 工具函数
    std::string Base64Encode(const std::string& input);
    std::string SHA1Hash(const std::string& input);
    bool SendWebSocketPing(int socket);
    bool CheckSocketHealth(int socket);
};

#endif // POSITION_SERVER_H