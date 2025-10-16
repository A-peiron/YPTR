#include "position_server.h"
#include "../utils/logging.h"
#include "../position/position_estimator.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <chrono>

// 定义静态常量成员
constexpr int PositionServer::MAX_FAILURES;
constexpr int PositionServer::PING_INTERVAL_SECONDS;
constexpr int PositionServer::HEALTH_CHECK_INTERVAL_MS;
constexpr int PositionServer::BROADCAST_INTERVAL_MS;
constexpr int PositionServer::CLIENT_TIMEOUT_SECONDS;

PositionServer::PositionServer(const std::string& host, int port)
    : host_(host), port_(port), is_running_(false), should_stop_(false) {
}

PositionServer::~PositionServer() {
    Stop();
}

bool PositionServer::Start() {
    if (is_running_) {
        NN_LOG_WARNING("Position server is already running");
        return true;
    }

    should_stop_ = false;

    // 启动主服务器线程
    server_thread_ = std::make_unique<std::thread>(&PositionServer::ServerThreadFunc, this);

    // 等待服务器启动
    int wait_count = 0;
    while (!is_running_ && wait_count < 100) {  // 最多等待1秒
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        wait_count++;
    }

    if (is_running_) {
        // 启动健康检查线程
        health_check_thread_ = std::make_unique<std::thread>(&PositionServer::HealthCheckThreadFunc, this);

        // 启动数据广播线程
        data_broadcast_thread_ = std::make_unique<std::thread>(&PositionServer::DataBroadcastThreadFunc, this);

        NN_LOG_INFO("Position server started on ws://%s:%d", host_.c_str(), port_);
        NN_LOG_INFO("Max clients: %d, Health check interval: %dms", max_clients_.load(), HEALTH_CHECK_INTERVAL_MS);
        return true;
    } else {
        NN_LOG_ERROR("Failed to start position server");
        return false;
    }
}

void PositionServer::Stop() {
    if (!is_running_) {
        return;
    }

    NN_LOG_INFO("Stopping position server...");
    should_stop_ = true;

    // 关闭所有客户端连接
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (auto& client : clients_) {
            if (client && client->socket_fd >= 0) {
                close(client->socket_fd);
                client->socket_fd = -1;
            }
        }
        clients_.clear();
    }

    // 等待所有线程结束
    if (server_thread_ && server_thread_->joinable()) {
        server_thread_->join();
    }

    if (health_check_thread_ && health_check_thread_->joinable()) {
        health_check_thread_->join();
    }

    if (data_broadcast_thread_ && data_broadcast_thread_->joinable()) {
        data_broadcast_thread_->join();
    }

    is_running_ = false;
    NN_LOG_INFO("Position server stopped");
}

void PositionServer::UpdatePositionData(const std::vector<PositionData>& positions) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    try {
        position_data_ = positions;

        if (!positions.empty()) {
            NN_LOG_INFO("Position server: Updated %zu person positions", positions.size());
        }

    } catch (const std::exception& e) {
        NN_LOG_ERROR("Exception in UpdatePositionData: %s", e.what());
        position_data_.clear();
    }
}

std::vector<PositionData> PositionServer::ExtractFromPersonPositions(
    const std::vector<PersonPosition>& person_positions,
    const std::vector<std::map<int, KeyPoint>>& keypoints) {

    std::vector<PositionData> positions;

    // 确保两个数组长度一致
    size_t min_size = std::min(person_positions.size(), keypoints.size());
    positions.reserve(min_size);

    for (size_t i = 0; i < min_size; ++i) {
        const PersonPosition& person = person_positions[i];

        if (person.valid) {
            // 归一化X坐标：从 ±244cm 映射到 ±1 (与参考项目一致)
            // 添加负号实现镜像效果（照镜子）
            float x_normalized = -(person.x_world * 100.0f / 244.0f); // 转换为cm再归一化，然后取反
            x_normalized = std::max(-1.0f, std::min(1.0f, x_normalized));

            PositionData pos(static_cast<int>(i), x_normalized, 0.0f, true);

            // 复制并归一化关键点坐标 (640x480分辨率)
            pos.keypoints = keypoints[i];
            for (auto& kp_pair : pos.keypoints) {
                KeyPoint& kp = kp_pair.second;
                // 归一化关键点坐标到 [0, 1] 范围
                kp.x = kp.x / 640.0f;  // 分辨率宽度640
                kp.y = kp.y / 480.0f;  // 分辨率高度480
            }

            positions.push_back(pos);
        }
    }

    return positions;
}

void PositionServer::ServerThreadFunc() {
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        NN_LOG_ERROR("Failed to create server socket");
        return;
    }

    // 设置socket选项
    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_);

    if (host_ == "0.0.0.0") {
        server_addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        inet_pton(AF_INET, host_.c_str(), &server_addr.sin_addr);
    }

    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        NN_LOG_ERROR("Failed to bind server socket to %s:%d", host_.c_str(), port_);
        close(server_socket);
        return;
    }

    if (listen(server_socket, 5) < 0) {
        NN_LOG_ERROR("Failed to listen on server socket");
        close(server_socket);
        return;
    }

    is_running_ = true;
    NN_LOG_INFO("Position server listening on %s:%d", host_.c_str(), port_);

    while (!should_stop_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(server_socket, &read_fds);

        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 100000; // 100ms timeout

        int select_result = select(server_socket + 1, &read_fds, nullptr, nullptr, &timeout);
        if (select_result > 0 && FD_ISSET(server_socket, &read_fds)) {
            int client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
            if (client_socket >= 0) {
                // 获取客户端IP
                std::string client_ip = GetClientIP(client_socket);

                // 在新线程中处理客户端连接
                std::thread client_thread(&PositionServer::HandleNewClient, this, client_socket, client_ip);
                client_thread.detach();
            }
        }
    }

    close(server_socket);
}

std::string PositionServer::GetClientIP(int socket) {
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);

    if (getpeername(socket, (struct sockaddr*)&addr, &addr_len) == 0) {
        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &addr.sin_addr, ip_str, INET_ADDRSTRLEN);
        return std::string(ip_str);
    }

    return "unknown";
}

void PositionServer::HealthCheckThreadFunc() {
    NN_LOG_INFO("Health check thread started");

    while (!should_stop_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(HEALTH_CHECK_INTERVAL_MS));

        if (should_stop_) break;

        // 清理断开的客户端
        CleanupDisconnectedClients();

        // 检查每个客户端的健康状态
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (auto& client : clients_) {
            if (client && client->is_active) {
                if (!ValidateClient(*client)) {
                    NN_LOG_WARNING("Client %s failed health check", client->client_ip.c_str());
                    client->is_active = false;
                }
            }
        }
    }

    NN_LOG_INFO("Health check thread stopped");
}

void PositionServer::DataBroadcastThreadFunc() {
    NN_LOG_INFO("Data broadcast thread started");

    while (!should_stop_) {
        try {
            std::this_thread::sleep_for(std::chrono::milliseconds(BROADCAST_INTERVAL_MS));

            if (should_stop_) break;

            // 创建JSON数据
            std::string json_data = CreatePositionDataJSON();

            // 广播给所有活跃客户端
            BroadcastToAllClients(json_data);

        } catch (const std::exception& e) {
            NN_LOG_ERROR("Exception in data broadcast thread: %s", e.what());
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // 等待1秒后重试
        } catch (...) {
            NN_LOG_ERROR("Unknown exception in data broadcast thread");
            std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // 等待1秒后重试
        }
    }

    NN_LOG_INFO("Data broadcast thread stopped");
}

void PositionServer::CleanupDisconnectedClients() {
    std::lock_guard<std::mutex> lock(clients_mutex_);

    auto it = std::remove_if(clients_.begin(), clients_.end(),
        [](const std::unique_ptr<ClientConnection>& client) {
            if (!client || !client->is_active) {
                if (client && client->socket_fd >= 0) {
                    close(client->socket_fd);
                }
                return true;
            }
            return false;
        });

    if (it != clients_.end()) {
        size_t removed = std::distance(it, clients_.end());
        clients_.erase(it, clients_.end());
        if (removed > 0) {
            NN_LOG_INFO("Cleaned up %zu disconnected clients", removed);
        }
    }
}

bool PositionServer::ValidateClient(ClientConnection& client) {
    if (client.socket_fd < 0) {
        return false;
    }

    // 检查socket是否仍然有效
    return CheckSocketHealth(client.socket_fd);
}

void PositionServer::BroadcastToAllClients(const std::string& data) {
    std::lock_guard<std::mutex> lock(clients_mutex_);

    for (auto& client : clients_) {
        if (client && client->is_active && client->socket_fd >= 0) {
            if (!SendWebSocketFrame(client->socket_fd, data)) {
                client->failure_count++;
                if (client->failure_count >= MAX_FAILURES) {
                    NN_LOG_WARNING("Client %s exceeded max failures, marking inactive",
                                 client->client_ip.c_str());
                    client->is_active = false;
                }
            } else {
                client->failure_count = 0;
                client->last_data_sent = std::chrono::steady_clock::now();
            }
        }
    }
}

bool PositionServer::CheckSocketHealth(int socket) {
    // 使用MSG_PEEK检查socket状态，不会真正读取数据
    char test_byte;
    int result = recv(socket, &test_byte, 1, MSG_PEEK | MSG_DONTWAIT);

    if (result == 0) {
        // 连接已关闭
        return false;
    } else if (result < 0) {
        // 检查错误类型
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // 没有数据可读，但连接正常
            return true;
        } else {
            // 其他错误，连接有问题
            return false;
        }
    }

    // 有数据可读，连接正常
    return true;
}

void PositionServer::HandleNewClient(int client_socket, const std::string& client_ip) {
    char buffer[4096];

    try {
        // 设置socket选项：超时
        struct timeval timeout;
        timeout.tv_sec = 5;  // 5秒超时
        timeout.tv_usec = 0;
        setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(client_socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        // 读取HTTP升级请求
        int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
        if (bytes_received <= 0) {
            NN_LOG_ERROR("Failed to receive WebSocket upgrade request from %s", client_ip.c_str());
            close(client_socket);
            return;
        }

        buffer[bytes_received] = '\0';
        std::string request(buffer);

        if (!IsWebSocketUpgrade(request)) {
            NN_LOG_WARNING("Invalid WebSocket upgrade request from %s", client_ip.c_str());
            close(client_socket);
            return;
        }

        // 提取WebSocket密钥并生成响应
        std::string key = ExtractWebSocketKey(request);
        if (key.empty()) {
            NN_LOG_ERROR("Missing WebSocket key in upgrade request from %s", client_ip.c_str());
            close(client_socket);
            return;
        }

        std::string response = GenerateWebSocketResponse(key);

        if (send(client_socket, response.c_str(), response.length(), 0) < 0) {
            NN_LOG_ERROR("Failed to send WebSocket upgrade response to %s", client_ip.c_str());
            close(client_socket);
            return;
        }

        // 检查客户端数量限制
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            if (clients_.size() >= static_cast<size_t>(max_clients_)) {
                NN_LOG_WARNING("Max clients reached, rejecting connection from %s", client_ip.c_str());
                close(client_socket);
                return;
            }
        }

        // 创建新的客户端连接对象
        auto new_client = std::make_unique<ClientConnection>(client_socket);
        new_client->client_ip = client_ip;
        new_client->is_active = true;

        // 添加到客户端列表
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            clients_.push_back(std::move(new_client));
        }

        NN_LOG_INFO("WebSocket client connected successfully from %s (total clients: %zu)",
                   client_ip.c_str(), clients_.size());

    } catch (const std::exception& e) {
        NN_LOG_ERROR("Exception in WebSocket client handler for %s: %s", client_ip.c_str(), e.what());
        close(client_socket);
    } catch (...) {
        NN_LOG_ERROR("Unknown exception in WebSocket client handler for %s", client_ip.c_str());
        close(client_socket);
    }
}

std::string PositionServer::GenerateWebSocketResponse(const std::string& key) {
    const std::string magic_string = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string concat = key + magic_string;
    std::string hash = SHA1Hash(concat);
    std::string encoded = Base64Encode(hash);

    std::ostringstream response;
    response << "HTTP/1.1 101 Switching Protocols\r\n"
             << "Upgrade: websocket\r\n"
             << "Connection: Upgrade\r\n"
             << "Sec-WebSocket-Accept: " << encoded << "\r\n"
             << "\r\n";

    return response.str();
}

std::string PositionServer::CreatePositionDataJSON() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    std::ostringstream json;
    json << "{\"balls\":[";

    bool first = true;

    for (const auto& person : position_data_) {
        if (!person.valid) continue;

        if (!first) {
            json << ",";
        }

        json << "{\"id\":" << person.person_id
             << ",\"x\":" << std::fixed << std::setprecision(3) << person.x_normalized
             << ",\"y\":" << std::fixed << std::setprecision(3) << person.y_normalized
             << ",\"stable\":" << (person.stable ? "true" : "false")
             << ",\"keypoints\":[";

        // 按YOLO-Pose关键点顺序(0-16)输出所有关键点
        bool first_kp = true;
        for (int kp_id = 0; kp_id < 17; ++kp_id) {
            if (!first_kp) {
                json << ",";
            }

            auto it = person.keypoints.find(kp_id);
            if (it != person.keypoints.end()) {
                // 关键点存在，输出归一化后的值
                const KeyPoint& kp = it->second;
                json << "{\"id\":" << kp_id
                     << ",\"x\":" << std::fixed << std::setprecision(3) << kp.x
                     << ",\"y\":" << std::fixed << std::setprecision(3) << kp.y
                     << ",\"score\":" << std::fixed << std::setprecision(3) << kp.score
                     << "}";
            } else {
                // 理论上不应该到这里，因为所有17个关键点都应该存在
                json << "{\"id\":" << kp_id
                     << ",\"x\":0"
                     << ",\"y\":0"
                     << ",\"score\":0.0"
                     << "}";
            }

            first_kp = false;
        }

        json << "]}";
        first = false;
    }

    json << "]}";

    return json.str();
}

bool PositionServer::SendWebSocketFrame(int socket, const std::string& data) {
    try {
        size_t data_length = data.length();
        std::vector<uint8_t> frame;

        // WebSocket frame格式
        frame.push_back(0x81); // FIN=1, opcode=1 (text)

        if (data_length < 126) {
            frame.push_back(static_cast<uint8_t>(data_length));
        } else if (data_length < 65536) {
            frame.push_back(126);
            frame.push_back(static_cast<uint8_t>(data_length >> 8));
            frame.push_back(static_cast<uint8_t>(data_length & 0xFF));
        } else {
            // 对于更大的数据，需要使用64位长度
            frame.push_back(127);
            for (int i = 7; i >= 0; i--) {
                frame.push_back(static_cast<uint8_t>((data_length >> (8 * i)) & 0xFF));
            }
        }

        // 添加数据
        frame.insert(frame.end(), data.begin(), data.end());

        // 发送frame - 使用分片发送处理大数据包
        size_t total_sent = 0;
        size_t chunk_size = 8192; // 8KB chunks

        while (total_sent < frame.size()) {
            size_t to_send = std::min(chunk_size, frame.size() - total_sent);
            ssize_t sent = send(socket, frame.data() + total_sent, to_send, MSG_NOSIGNAL);

            if (sent < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    // 暂时不可写，等待一下再重试
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                } else {
                    NN_LOG_ERROR("Send error: %s", strerror(errno));
                    return false;
                }
            } else if (sent == 0) {
                NN_LOG_WARNING("Connection closed by peer during send");
                return false;
            }

            total_sent += sent;
        }

        return total_sent == frame.size();

    } catch (const std::exception& e) {
        NN_LOG_ERROR("Exception in SendWebSocketFrame: %s", e.what());
        return false;
    } catch (...) {
        NN_LOG_ERROR("Unknown exception in SendWebSocketFrame");
        return false;
    }
}

bool PositionServer::IsWebSocketUpgrade(const std::string& request) {
    return request.find("Upgrade: websocket") != std::string::npos;
}

std::string PositionServer::ExtractWebSocketKey(const std::string& request) {
    const std::string key_prefix = "Sec-WebSocket-Key: ";
    size_t start = request.find(key_prefix);
    if (start == std::string::npos) return "";

    start += key_prefix.length();
    size_t end = request.find("\r\n", start);
    if (end == std::string::npos) return "";

    return request.substr(start, end - start);
}

std::string PositionServer::Base64Encode(const std::string& input) {
    BIO *bio, *b64;
    BUF_MEM *buffer_ptr;

    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(bio, input.c_str(), input.length());
    BIO_flush(bio);
    BIO_get_mem_ptr(bio, &buffer_ptr);

    std::string result(buffer_ptr->data, buffer_ptr->length);
    BIO_free_all(bio);

    return result;
}

std::string PositionServer::SHA1Hash(const std::string& input) {
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(reinterpret_cast<const unsigned char*>(input.c_str()), input.length(), hash);
    return std::string(reinterpret_cast<char*>(hash), SHA_DIGEST_LENGTH);
}

bool PositionServer::SendWebSocketPing(int socket) {
    try {
        // WebSocket Ping frame (opcode = 0x9)
        std::vector<uint8_t> ping_frame = {0x89, 0x00}; // FIN=1, opcode=9 (ping), payload length=0

        ssize_t sent = send(socket, ping_frame.data(), ping_frame.size(), MSG_NOSIGNAL);
        return sent == static_cast<ssize_t>(ping_frame.size());

    } catch (const std::exception& e) {
        NN_LOG_ERROR("Exception in SendWebSocketPing: %s", e.what());
        return false;
    } catch (...) {
        NN_LOG_ERROR("Unknown exception in SendWebSocketPing");
        return false;
    }
}