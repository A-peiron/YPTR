#include "ReConfig.h"
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <map>
#include <sstream>
#include "types.h"

namespace rr
{

	bool RrConfig::IsSpace(char c)
	{
		if (' ' == c || '\t' == c)
			return true;
		return false;
	}

	bool RrConfig::IsCommentChar(char c)
	{
		switch (c)
		{
		case '#':
			return true;
		default:
			return false;
		}
	}

	void RrConfig::Trim(std::string &str)
	{
		if (str.empty())
		{
			return;
		}
		int i, start_pos, end_pos;
		for (i = 0; i < str.size(); ++i)
		{
			if (!IsSpace(str[i]))
			{
				break;
			}
		}
		if (i == str.size())
		{
			str = "";
			return;
		}
		start_pos = i;
		for (i = str.size() - 1; i >= 0; --i)
		{
			if (!IsSpace(str[i]))
			{
				break;
			}
		}
		end_pos = i;
		str = str.substr(start_pos, end_pos - start_pos + 1);
	}

	bool RrConfig::AnalyseLine(const std::string &line, std::string &section, std::string &key, std::string &value)
	{
		if (line.empty())
			return false;
		int start_pos = 0, end_pos = line.size() - 1, pos, s_startpos, s_endpos;
		if ((pos = line.find("#")) != -1)
		{
			if (0 == pos)
			{
				return false;
			}
			end_pos = pos - 1;
		}
		if (((s_startpos = line.find("[")) != -1) && ((s_endpos = line.find("]"))) != -1)
		{
			section = line.substr(s_startpos + 1, s_endpos - 1);
			return true;
		}
		std::string new_line = line.substr(start_pos, start_pos + 1 - end_pos);
		if ((pos = new_line.find('=')) == -1)
			return false;
		key = new_line.substr(0, pos);
		value = new_line.substr(pos + 1, end_pos + 1 - (pos + 1));
		Trim(key);
		if (key.empty())
		{
			return false;
		}
		Trim(value);
		if ((pos = value.find("\r")) > 0)
		{
			value.replace(pos, 1, "");
		}
		if ((pos = value.find("\n")) > 0)
		{
			value.replace(pos, 1, "");
		}
		return true;
	}

	bool RrConfig::ReadConfig(const std::string &filename)
	{
		settings_.clear();
		std::ifstream infile(filename.c_str()); // 构造默认调用open,所以可以不调用open
		// std::ifstream infile;
		// infile.open(filename.c_str());
		// bool ret = infile.is_open()
		if (!infile)
		{
			return false;
		}
		std::string line, key, value, section;
		std::map<std::string, std::string> k_v;
		std::map<std::string, std::map<std::string, std::string>>::iterator it;
		while (getline(infile, line))
		{
			if (AnalyseLine(line, section, key, value))
			{
				it = settings_.find(section);
				if (it != settings_.end())
				{
					k_v[key] = value;
					it->second = k_v;
				}
				else
				{
					k_v.clear();
					settings_.insert(std::make_pair(section, k_v));
				}
			}
			key.clear();
			value.clear();
		}
		infile.close();
		return true;
	}

	std::string RrConfig::ReadString(const char *section, const char *item, const char *default_value)
	{
		std::string tmp_s(section);
		std::string tmp_i(item);
		std::string def(default_value);
		std::map<std::string, std::string> k_v;
		std::map<std::string, std::string>::iterator it_item;
		std::map<std::string, std::map<std::string, std::string>>::iterator it;
		it = settings_.find(tmp_s);
		if (it == settings_.end())
		{
			return def;
		}
		k_v = it->second;
		it_item = k_v.find(tmp_i);
		if (it_item == k_v.end())
		{
			return def;
		}
		return it_item->second;
	}

	int RrConfig::ReadInt(const char *section, const char *item, const int &default_value)
	{
		std::string tmp_s(section);
		std::string tmp_i(item);
		std::map<std::string, std::string> k_v;
		std::map<std::string, std::string>::iterator it_item;
		std::map<std::string, std::map<std::string, std::string>>::iterator it;
		it = settings_.find(tmp_s);
		if (it == settings_.end())
		{
			return default_value;
		}
		k_v = it->second;
		it_item = k_v.find(tmp_i);
		if (it_item == k_v.end())
		{
			return default_value;
		}
		return atoi(it_item->second.c_str());
	}

	float RrConfig::ReadFloat(const char *section, const char *item, const float &default_value)
	{
		std::string tmp_s(section);
		std::string tmp_i(item);
		std::map<std::string, std::string> k_v;
		std::map<std::string, std::string>::iterator it_item;
		std::map<std::string, std::map<std::string, std::string>>::iterator it;
		it = settings_.find(tmp_s);
		if (it == settings_.end())
		{
			return default_value;
		}
		k_v = it->second;
		it_item = k_v.find(tmp_i);
		if (it_item == k_v.end())
		{
			return default_value;
		}
		return atof(it_item->second.c_str());
	}

	/**
	 * 这个方法使用来特殊处理我自己定义的配置使用的
	 * 格式要求{key,0.8}${key2,0.8}
	 */
	std::map<std::string, double> RrConfig::ReadStringToMap(const char *section, const char *item)
	{
		std::string tmp_s(section);
		std::string tmp_i(item);
		std::map<std::string, std::string> k_v;
		std::map<std::string, std::string>::iterator it_item;
		std::map<std::string, std::map<std::string, std::string>>::iterator it;
		it = settings_.find(tmp_s);
		k_v = it->second;
		it_item = k_v.find(tmp_i);
		std::map<std::string, double> result;
		std::string pairString;
		std::istringstream iss(it_item->second);
		// 按照"$"分割字符串
		while (std::getline(iss, pairString, '$'))
		{
			std::size_t start = pairString.find('{');	 // 查找起始位置
			std::size_t end = pairString.find('}');		 // 查找结束位置
			std::size_t commaPos = pairString.find(','); // 查找逗号位置
			if (start != std::string::npos && end != std::string::npos && commaPos != std::string::npos)
			{

				std::string key = pairString.substr(start + 1, commaPos - start - 1);	   // 提取键
				double value = std::stod(pairString.substr(commaPos + 1, end - commaPos)); // 提取值

				result[key] = value; // 添加键值对到map中
			}
		}

		return result;
	}
	/**
	 * 读取范围点信息
	*/
	void RrConfig::readPoints(std::string filename, Polygon &g_ploygon, int width, int height)
	{
		std::ifstream file(filename);
		std::string str;
		while (std::getline(file, str))
		{
			std::stringstream ss(str);
			std::string x, y;
			std::getline(ss, x, ',');
			std::getline(ss, y, ',');

			// recover to original size
			x = std::to_string(std::stof(x) * width);
			y = std::to_string(std::stof(y) * height);

			g_ploygon.push_back({std::stoi(x), std::stoi(y)});
		}
	}

	/**
	 * 获取时间段
	*/
	std::vector<TimeRange> RrConfig::ReadStringToTimeRanges(const char *section, const char *item, const char* default_value)
	{
		std::string tmp_s(section);
		std::string tmp_i(item);

		std::map<std::string, std::string> k_v;
		std::map<std::string, std::string>::iterator it_item;
		std::map<std::string, std::map<std::string, std::string>>::iterator it;

		// 在 settings_ 中查找指定的部分
		it = settings_.find(tmp_s);
		std::vector<TimeRange> result;
		std::string rangeString;
		std::istringstream iss;

		k_v = it->second;
		it_item = k_v.find(tmp_i);
		if (it_item == k_v.end())
			iss = std::istringstream(default_value);
		else
			iss = std::istringstream(it_item->second);

		while (std::getline(iss, rangeString, '$'))
		{
			TimeRange range;
			char discard;
			std::istringstream range_iss(rangeString);
			range_iss >> discard >> range.startHour >> discard >> range.startMinute >> discard 
                  >> range.endHour >> discard >> range.endMinute >> discard;
			result.push_back(range);
		}

		return result;
	}
}

/**
 * 判断当前时间是否在规定时间之内
*/
bool IsCurrentTimeInRange(int startHour, int startMinute, int endHour, int endMinute) {
    // 获取当前时间
    std::time_t currentTime = std::time(nullptr);
    std::tm* currentLocalTime = std::localtime(&currentTime);

    // 设置规定时间的开始时间
    std::tm startTime = *currentLocalTime;
    startTime.tm_hour = startHour;
    startTime.tm_min = startMinute;
    std::time_t startTimestamp = std::mktime(&startTime);

    // 设置规定时间的结束时间
    std::tm endTime = *currentLocalTime;
    endTime.tm_hour = endHour;
    endTime.tm_min = endMinute;
    std::time_t endTimestamp = std::mktime(&endTime);

    // 获取当前时间的时间戳
    std::time_t currentTimestamp = std::mktime(currentLocalTime);

    // 判断当前时间是否在规定时间范围内
    return (currentTimestamp >= startTimestamp && currentTimestamp <= endTimestamp);
}

/**
 * 判断当前时间是否在规定的多个时间段之内
*/
bool IsCurrentTimeInRanges(const std::vector<TimeRange>& timeRanges) {
    // 遍历所有时间段，判断当前时间是否在任何一个时间段内
    for (const auto& range : timeRanges) {
        if (IsCurrentTimeInRange(range.startHour, range.startMinute, range.endHour, range.endMinute)) {
            return true;
        }
    }

    // 当前时间不在任何一个时间段内
    return false;
}