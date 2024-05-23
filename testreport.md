## 注册
以下是根据 `UserServiceImpl` 测试类生成的测试报告，内容包括用例ID、优先级、接口（类）名称、用例名、前置条件、请求参数、接口预期返回内容：

### 测试报告

| 用例ID | 优先级 (Pn) | 接口（类）名称         | 用例名                                           | 前置条件                      | 请求参数                          | 接口预期返回内容                             |
|-------|-------------|------------------------|--------------------------------------------------|-------------------------------|-----------------------------------|---------------------------------------------|
| TC011 | P1          | UserServiceImpl         | testRegisterUser_Success                         | 用户名不存在                     | UserRegisterDTO                   | 成功注册用户，并插入用户信息到数据库               |
| TC012 | P1          | UserServiceImpl         | testRegisterUser_UserAlreadyExists               | 用户名已存在                     | UserRegisterDTO                   | 抛出BusinessException，错误代码为USER_ALREADY_EXIST |

### 测试报告详情

#### 用例ID: TC011
- **优先级 (Pn)**: P1
- **接口（类）名称**: UserServiceImpl
- **用例名**: testRegisterUser_Success
- **前置条件**: 用户名不存在
- **请求参数**:
  - UserRegisterDTO:
    - username: "testUser"
    - realName: "Test User"
    - password: "testPass"
- **接口预期返回内容**: 成功注册用户，并插入用户信息到数据库

#### 用例ID: TC012
- **优先级 (Pn)**: P1
- **接口（类）名称**: UserServiceImpl
- **用例名**: testRegisterUser_UserAlreadyExists
- **前置条件**: 用户名已存在
- **请求参数**:
  - UserRegisterDTO:
    - username: "existingUser"
    - realName: "Existing User"
    - password: "testPass"
- **接口预期返回内容**: 抛出BusinessException，错误代码为USER_ALREADY_EXIST

### 结论
以上测试用例涵盖了 `UserServiceImpl` 类的主要功能，确保了用户注册时用户名是否存在的验证以及成功注册的处理逻辑。测试报告为中文，详细描述了每个测试用例的ID、优先级、接口名称、用例名、前置条件、请求参数和接口预期返回内容。


## 导出标注

以下是根据 `ResultExportServiceImpl` 测试类生成的测试报告，内容包括用例ID、优先级、接口（类）名称、用例名、前置条件、请求参数、接口预期返回内容：

### 测试报告

| 用例ID | 优先级 (Pn) | 接口（类）名称         | 用例名                                           | 前置条件                      | 请求参数                          | 接口预期返回内容                             |
|-------|-------------|------------------------|--------------------------------------------------|-------------------------------|-----------------------------------|---------------------------------------------|
| TC009 | P1          | ResultExportServiceImpl | export_whenTagKeyNotUnique_shouldThrowBusinessException | 标签键不唯一                      | projectId                        | 抛出BusinessException，错误代码为DUPLICATE_TAG_KEY |
| TC010 | P1          | ResultExportServiceImpl | export_shouldExportResultsSuccessfully             | 项目存在且标签键唯一，传入有效的projectId | projectId                        | 成功导出标注结果，并返回结果列表                   |

### 测试报告详情

#### 用例ID: TC009
- **优先级 (Pn)**: P1
- **接口（类）名称**: ResultExportServiceImpl
- **用例名**: export_whenTagKeyNotUnique_shouldThrowBusinessException
- **前置条件**: 标签键不唯一
- **请求参数**:
  - projectId: "project1"
- **接口预期返回内容**: 抛出BusinessException，错误代码为DUPLICATE_TAG_KEY

#### 用例ID: TC010
- **优先级 (Pn)**: P1
- **接口（类）名称**: ResultExportServiceImpl
- **用例名**: export_shouldExportResultsSuccessfully
- **前置条件**: 项目存在且标签键唯一，传入有效的projectId
- **请求参数**:
  - projectId: "project1"
- **接口预期返回内容**: 成功导出标注结果，并返回结果列表

### 结论
以上测试用例涵盖了 `ResultExportServiceImpl` 类的主要功能，确保了导出标注结果时标签键唯一性的验证以及成功导出的处理逻辑。测试报告为中文，详细描述了每个测试用例的ID、优先级、接口名称、用例名、前置条件、请求参数和接口预期返回内容。

# 上传论文

以下是根据 `PaperServiceImpl` 测试类生成的测试报告，内容包括用例ID、优先级、接口（类）名称、用例名、前置条件、请求参数、接口预期返回内容：

### 测试报告

| 用例ID | 优先级 (Pn) | 接口（类）名称 | 用例名 | 前置条件 | 请求参数 | 接口预期返回内容 |
|-------|-------------|----------------|--------|----------|----------|------------------|
| TC007 | P1          | PaperServiceImpl | upload_whenProjectDoesNotExist_shouldThrowBusinessException | 传入的projectId不存在 | projectId, MultipartFile数组 | 抛出BusinessException，错误代码为PROJECT_NOT_EXIST |
| TC008 | P1          | PaperServiceImpl | upload_shouldUploadPaperSuccessfully | 项目存在且已分配，传入有效的MultipartFile数组 | projectId, MultipartFile数组 | 成功上传文件，触发文件转换并返回结果 |

### 测试报告详情

#### 用例ID: TC007
- **优先级 (Pn)**: P1
- **接口（类）名称**: PaperServiceImpl
- **用例名**: upload_whenProjectDoesNotExist_shouldThrowBusinessException
- **前置条件**: 传入的projectId不存在
- **请求参数**:
  - projectId: "无效的项目ID"
  - MultipartFile数组: [multipartFile]
- **接口预期返回内容**: 抛出BusinessException，错误代码为PROJECT_NOT_EXIST

#### 用例ID: TC008
- **优先级 (Pn)**: P1
- **接口（类）名称**: PaperServiceImpl
- **用例名**: upload_shouldUploadPaperSuccessfully
- **前置条件**: 项目存在且已分配，传入有效的MultipartFile数组
- **请求参数**:
  - projectId: "project1"
  - MultipartFile数组: [multipartFile]
- **接口预期返回内容**: 成功上传文件，触发文件转换并返回结果

### 结论
以上测试用例涵盖了 `PaperServiceImpl` 类的主要功能，确保了上传文件时项目是否存在的验证以及文件上传成功的处理逻辑。测试报告为中文，详细描述了每个测试用例的ID、优先级、接口名称、用例名、前置条件、请求参数和接口预期返回内容。

# 标注统计

以下是根据 `AnnotatorServiceImpl` 测试类生成的测试报告，内容包括用例ID、优先级、接口（类）名称、用例名、前置条件、请求参数、接口预期返回内容：

### 测试报告

| 用例ID | 优先级 (Pn) | 接口（类）名称 | 用例名 | 前置条件 | 请求参数 | 接口预期返回内容 |
|-------|-------------|----------------|--------|----------|----------|------------------|
| TC001 | P1          | AnnotatorServiceImpl | batchAddAnnotator_shouldAddAnnotatorsSuccessfully | 项目已分配，传入有效的userIds列表 | projectId, userIds | 成功添加标注员，并返回所有标注员的ID列表 |
| TC002 | P1          | AnnotatorServiceImpl | batchRemoveAnnotator_shouldRemoveAnnotatorsSuccessfully | 项目已分配，传入有效的userIds列表 | projectId, userIds | 成功移除标注员，并返回剩余的标注员ID列表 |
| TC003 | P2          | AnnotatorServiceImpl | pageAnnotator_shouldReturnPageModel | 项目存在，传入有效的分页参数 | projectId, pageNum, pageSize | 返回分页的标注员用户列表 |
| TC004 | P1          | AnnotatorServiceImpl | getAllAnnotatorIds_shouldReturnAllAnnotatorIds | 项目存在 | projectId | 返回项目中所有标注员的ID列表 |
| TC005 | P2          | AnnotatorServiceImpl | countAnnotator_shouldReturnAnnotatorCount | 项目存在 | projectId | 返回项目中标注员的总数 |
| TC006 | P2          | AnnotatorServiceImpl | listAnnotatorByProject_shouldReturnListOfAnnotators | 项目存在 | projectId | 返回项目中的标注员列表 |

### 测试报告详情

#### 用例ID: TC001
- **优先级 (Pn)**: P1
- **接口（类）名称**: AnnotatorServiceImpl
- **用例名**: batchAddAnnotator_shouldAddAnnotatorsSuccessfully
- **前置条件**: 项目已分配，传入有效的userIds列表
- **请求参数**:
  - projectId: "project1"
  - userIds: ["user1", "user2", "user3"]
- **接口预期返回内容**: 成功添加标注员，并返回所有标注员的ID列表

#### 用例ID: TC002
- **优先级 (Pn)**: P1
- **接口（类）名称**: AnnotatorServiceImpl
- **用例名**: batchRemoveAnnotator_shouldRemoveAnnotatorsSuccessfully
- **前置条件**: 项目已分配，传入有效的userIds列表
- **请求参数**:
  - projectId: "project1"
  - userIds: ["user1", "user2", "user3"]
- **接口预期返回内容**: 成功移除标注员，并返回剩余的标注员ID列表

#### 用例ID: TC003
- **优先级 (Pn)**: P2
- **接口（类）名称**: AnnotatorServiceImpl
- **用例名**: pageAnnotator_shouldReturnPageModel
- **前置条件**: 项目存在，传入有效的分页参数
- **请求参数**:
  - projectId: "project1"
  - pageNum: 1
  - pageSize: 10
- **接口预期返回内容**: 返回分页的标注员用户列表

#### 用例ID: TC004
- **优先级 (Pn)**: P1
- **接口（类）名称**: AnnotatorServiceImpl
- **用例名**: getAllAnnotatorIds_shouldReturnAllAnnotatorIds
- **前置条件**: 项目存在
- **请求参数**:
  - projectId: "project1"
- **接口预期返回内容**: 返回项目中所有标注员的ID列表

#### 用例ID: TC005
- **优先级 (Pn)**: P2
- **接口（类）名称**: AnnotatorServiceImpl
- **用例名**: countAnnotator_shouldReturnAnnotatorCount
- **前置条件**: 项目存在
- **请求参数**:
  - projectId: "project1"
- **接口预期返回内容**: 返回项目中标注员的总数

#### 用例ID: TC006
- **优先级 (Pn)**: P2
- **接口（类）名称**: AnnotatorServiceImpl
- **用例名**: listAnnotatorByProject_shouldReturnListOfAnnotators
- **前置条件**: 项目存在
- **请求参数**:
  - projectId: "project1"
- **接口预期返回内容**: 返回项目中的标注员列表

### 结论
以上测试用例涵盖了 `AnnotatorServiceImpl` 类的主要功能，确保了添加、移除、分页获取、获取所有标注员ID、统计标注员数量和列出项目中的标注员等功能的正确性。测试报告为中文，详细描述了每个测试用例的ID、优先级、接口名称、用例名、前置条件、请求参数和接口预期返回内容。